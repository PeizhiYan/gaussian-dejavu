#
# Copyright (C) 2024, DejaVu Authors 
# The copyright applies to the modified code segments.
#

#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid
# from pytorch3d.transforms import quaternion_multiply
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.graphics_utils import LitePointCloud


class LiteGaussianModel:
    """A light-weight version of 3D Gaussian model"""
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, device: str):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.device = device   # computing device
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.binding,
            self.binding_counter,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def create_from_pcd(self, pcd: LitePointCloud, spatial_lr_scale: float = 1.0):
        """This function is for demonstration only"""
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        self._xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = torch.nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        print("Number of points at initialisation: ", self.get_xyz.shape[0])

        dist2 = torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001) 
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = torch.nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        
        self._xyz = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = torch.nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = torch.nn.Parameter(scales.requires_grad_(True))
        self._rotation = torch.nn.Parameter(rots.requires_grad_(True))
        self._opacity = torch.nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        


