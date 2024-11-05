######################################
## GaussianDejavu Framework Base     #
## Author: Peizhi Yan                #
##   Date: 03/27/2024                #
## Update: 10/31/2024                #
######################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import cv2
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import Resize

from utils.uv_rasterizer import UV_Rasterizer
from networks import EDNet
#from models.framework import img2tensor

from utils.scene import Scene
from utils.scene.lite_gaussian_model import LiteGaussianModel
from utils.scene.uv_gaussian_model import UVGaussianModel, mesh_2_uv
from utils.scene.cameras import PerspectiveCamera, prepare_camera
from utils.viewer_utils import OrbitCamera
from utils.arguments import ModelParams, PipelineParams, OptimizationParams

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.gaussian_renderer import render

#import utils.o3d_utils as o3d_utils
from utils.image_utils import read_img, min_max, uint8_img, norm_img, image_align, display_landmarks_with_cv2
from utils.general_utils import dict2obj
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.graphics_utils import LitePointCloud, create_diff_world_to_view_matrix, verts_clip_to_ndc
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2


def img2tensor(img_tensor):
    """
    input:
    - img_tensor: torch.tensor  uint8  [N,H,W,C]   0 ~ 255
    output:
    - output: torch.tensor    float32  [N,C,H,W]   -1.0 ~ 1.0
    """
    output = img_tensor.permute(0,3,1,2)  # convert shape to NCHW
    output = output.type(torch.float32)   # convert dtype to float32
    output = output / 127.5 - 1           # normalize to -1.0 ~ 1.0
    return output


class Framework():
    """
    Version: DejaVu-V1 (v3.1)
    The framework that takes the reconstructed FLAME meshes and the original images,
    to create the 3D head Gaussians and render them to images.
    """

    def __init__(self, device, uv_rasterizer_device='cpu', uv_size=320):
        @dataclass
        class PipelineConfig:
            # Gaussian splatting rendering pipeline configuration
            debug: bool = True
            compute_cov3D_python: bool = False
            convert_SHs_python: bool = False

        self.version = '3.1'
        self.device = device
        self.uv_size = uv_size  # S
        self.uv_rasterizer_device = uv_rasterizer_device # it can be another CUDA device other than self.device
        self.network = EDNet(in_channels=3, out_channels=13, 
                             img_size=320, upmethod='transpose').to(self.device)
        self.resize_uv_map = Resize((self.uv_size, self.uv_size))
        self.uv_rasterizer = UV_Rasterizer(obj_path='./models/head_template.obj', 
                                           uv_size=self.uv_size, 
                                           device=self.uv_rasterizer_device)
        self.uv_init_opacity = np.load('./models/uv_init_opacity_weights.npy') # [256,256]
        self.uv_init_opacity = cv2.resize(self.uv_init_opacity, (self.uv_size, self.uv_size)) # [S,S]
        self.uv_mouth_interior_mask = torch.from_numpy(1 - self.uv_rasterizer.uv_mask_mouth_interior).to(device) # [S,S]
        self.uv_mouth_interior_mask = self.uv_mouth_interior_mask.bool()

        # Gaussian Splatting Renderer settings
        self.set_render_size(512) # default render size is 512x512, can change outside
        self.fov = 20.0  # x&y-axis FOV
        self.bg_color = (1.0,1.0,1.0)
        self.znear = 0.01
        self.zfar  = 100.0
        self.pipeline = PipelineConfig

        # Upper and Lower teeth indices
        self.upper_teeth_indices = [1725, 1743, 1663, 1662, 1661, 3547, 2778, 2779, 2780, 2858, 2842, 2841,
                                    1745, 1741, 1664, 1659, 1660, 3549, 2777, 2776, 2781, 2856, 2860, 2859]
        self.lower_teeth_indices = [1765, 1781, 1780, 1779, 1847, 1846, 3512, 2935, 2936, 2886, 2887, 2888, 2873, 
                                    1854, 1861, 1831, 1832, 1851, 3500, 2940, 2932, 2931, 2944, 2942]

        print(f'Framework v{self.version} initialized.')

    def set_render_size(self, render_size=512):
        self.H = self.W = render_size

    def create_init_uv_maps(self, 
                            batch_vertices : np.array, 
                            mouth_interior: bool = True):
        batch_size = len(batch_vertices)

        # Add teeth to the input shapes
        batch_vertices = np.copy(batch_vertices)
        batch_vertices[:,self.upper_teeth_indices,1] -= 0.002
        #batch_vertices[:,self.lower_teeth_indices,1] += 0.008

        # Create initial UV Gaussians maps
        uv_maps = np.zeros([batch_size, self.uv_size, self.uv_size, 13], dtype=np.float32)
        for i in range(batch_size):
            uv_map = mesh_2_uv(self.uv_rasterizer, batch_vertices[i], colors=None, compute_scales=False, mouth_interior=mouth_interior)
            uv_maps[i] = uv_map
        with torch.no_grad():
            uv_maps[:,:,:,9] = self.uv_init_opacity
            uv_maps = torch.from_numpy(uv_maps).to(self.device) # [N, S, S, 13]

        return uv_maps

    def get_uv_offsets(self, batch_images : torch.tensor, uv_maps : torch.tensor):
        with torch.no_grad():
            _batch_images_ = img2tensor(batch_images)  # [N, 3, H, W]   float32  -1 ~ 1
            _uv_positions_ = uv_maps.permute(0,3,1,2)[:,:3,:,:]   # [N, 3, S, S]
        fmaps, uv_offsets = self.network(_batch_images_, _uv_positions_)
        uv_offsets = self.resize_uv_map(uv_offsets)
        fmaps = fmaps.permute(0,2,3,1) # [N, Sf, Sf, F]
        uv_offsets = uv_offsets.permute(0,2,3,1) # [N, S, S, 13]
        return fmaps, uv_offsets

    def add_uv_offsts(self, uv_maps : torch.tensor, uv_offsets : torch.tensor, loc_discount : int):
        mouth_mask = self.uv_mouth_interior_mask.unsqueeze(0).unsqueeze(-1) # [N,S,S,3]
        uv_offsets[:,:,:,:3] = uv_offsets[:,:,:,:3] * mouth_mask # mask out the location offsets of mouth interior
        uv_offsets[:,:,:,:3] *= loc_discount
        uv_maps_new = uv_maps + uv_offsets
        return uv_maps_new
    
    def uv_maps_2_gaussians(self, uv_maps):
        batch_size = len(uv_maps)
        batch_gaussians = []
        for i in range(batch_size):
            gaussians = UVGaussianModel(uv_rasterizer=self.uv_rasterizer, device=self.device)
            gaussians.create_from_uv(uv_maps[i])
            batch_gaussians.append(gaussians)
        return batch_gaussians

    def create_batch_gaussians(self, 
                               batch_vertices : np.array,  
                               batch_images : torch.tensor, 
                               loc_discount : int = 1.0,
                               mouth_interior: bool = True,
                               return_all : bool = False
                               ):
        # - batch_vertices: [N, 5023, 3]    numpy.array
        # - batch_images:   [N, H, W, 3]    torch.tensor  uint8    0 ~ 255
        # - loc_discount:   location offset discount
        # - return_all:   if True, return the feature maps, UV maps and offsets
        # returns: batch of UVGaussian objects

        # 1. Create initial UV Gaussians maps
        uv_maps = self.create_init_uv_maps(batch_vertices, mouth_interior) # [N, S, S, 13]
        
        # 2. Get UV offsets from unet model
        fmaps, uv_offsets = self.get_uv_offsets(batch_images, uv_maps)

        # 3. Add UV offsets to initial uv maps
        uv_maps_new = self.add_uv_offsts(uv_maps, uv_offsets, loc_discount)

        # 4. Convert UV maps to UV Gaussians
        batch_gaussians = self.uv_maps_2_gaussians(uv_maps=uv_maps_new)

        if return_all:
            return batch_gaussians, fmaps, uv_maps, uv_offsets        
        else:
            return batch_gaussians

    def render_batch_gaussians(self, batch_cam_poses, batch_gaussians):
        # - batch_cam_poses: [n, 6]  torch.tensor  the 6DoF camera poses  
        # returns rendered images [N, H, W, 3]   torch.tensor

        batch_size = len(batch_cam_poses)

        batch_rendered = []
        for i in range(batch_size):
            gaussians = batch_gaussians[i]
            camera_pose = batch_cam_poses[i]
            Rt = create_diff_world_to_view_matrix(camera_pose)
            cam = PerspectiveCamera(Rt=Rt, fov=self.fov, bg=self.bg_color, image_width=self.W, image_height=self.H, 
                                    znear=self.znear, zfar=self.zfar)
            camera = prepare_camera(cam, self.device)
            output = render(viewpoint_camera=camera, pc=gaussians, pipe=self.pipeline, 
                            bg_color=torch.tensor(self.bg_color).to(self.device), scaling_modifier=1.0)
            rendered = output['render'] # [C,H,W]
            batch_rendered.append(rendered)
        
        # stack the rendered images to get the tensor of [N,C,H,W]
        batch_rendered = torch.stack(batch_rendered, dim=0)

        return batch_rendered




