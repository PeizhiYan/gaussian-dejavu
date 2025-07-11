######################################
## GaussianDejavu Framework          #
## Author: Peizhi Yan                #
##   Date: 11/04/2024                #
## Update: 06/18/2025                #
######################################

import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
import math
from torch.optim.lr_scheduler import StepLR

# Dejavu
from dejavu_base import Framework
from networks import MLP
from utils.loss import *
from utils.scene.cameras import PerspectiveCamera, prepare_camera
from utils.viewer_utils import OrbitCamera
from utils.gaussian_renderer import render
from utils.graphics_utils import create_diff_world_to_view_matrix, verts_clip_to_ndc
from utils.personal_video_utils import *
from utils.loss_utils import *

## FLAME
from utils.flame_lib import dict2obj
from utils.flame_lib.FLAME import FLAME


def load_model_weights(model, weights_path):
    try:
        model.load_state_dict(torch.load(weights_path), strict=False)
        print('model loaded from: ', weights_path)
    except:
        print('cannot load from: ', weights_path)
    return model



class GaussianDejavu():
    """
    Gaussian DejaVu Head Avatar Framework v1.3
    """

    def __init__(self, network_weights='./models/dejavu_network.pt', uv_map_size=320):

        device = "cuda:0"  # Don't use cuda:1, it causes illegal cuda memory access error, need to fix this bug later.\
        self.device = device
        uv_rasterization_device = device # this can be on other CUDA devices, since we do not backpropagate through it
        flame_device = device
        self.num_expressions = 10 # we use 10 expression blendmaps

        # FLAME Model
        flame_cfg = {
            'flame_model_path': './models/FLAME2020/generic_model.pkl',
            'flame_lmk_embedding_path': './models/landmark_embedding.npy',
            'camera_params': 3,
            'shape_params': 100,
            'expression_params': 50,
            'pose_params': 6,
            'tex_params': 50,
            'use_face_contour': False, 
            'cropped_size': 256,
            'batch_size': 1,
            'image_size': 224,
            'e_lr': 0.01,
            'e_wd': 0.0001,
            'savefolder': './test_results/',
            # weights of losses and reg terms
            'w_pho': 8,
            'w_lmks': 1,
            'w_shape_reg': 1e-4,
            'w_expr_reg': 1e-4,
            'w_pose_reg': 0,
        }
        # Create FLAME model
        flame_cfg = dict2obj(flame_cfg)
        self.flame = FLAME(flame_cfg).to(flame_device)
        # self.flame.v_template[3931:5023, 2] -= 0.005 # move the eyeballs backward a little bit

        # Create DejaVu base framework
        self.framework = Framework(device=device, uv_rasterizer_device=uv_rasterization_device, uv_size=uv_map_size)
        self.framework.network = self.framework.network.eval()
        self.framework.set_render_size(512)
        print('Number of Gaussians: ', len(self.framework.uv_rasterizer.valid_coords[0]))

        # Load the trained network weights
        self.framework.network = load_model_weights(self.framework.network, weights_path=network_weights)

        # Initialize UV regularization weights
        self._init_uv_regularization_weights_()

        # Initialize head avatar parameters
        self.mean_uv_offsets = None
        self.mean_shape_coefficients = None
        self.mean_shape = None
        self.mean_exp_coefficients = None
        self.global_uv_delta = None
        self.uv_delta_blendmaps = None

        # Create the MLP network for mapping expression coefficients + jaw pose to blending weights
        self.mlp = MLP(input_size=50 + 3, hidden_size=50, output_size=self.num_expressions).to(device)
        self.mlp.load_state_dict(torch.load(os.path.join('./models/mediapipe_to_flame/mlp.pth')))

        print('Gaussian DejaVu Framework Created.')


    def _init_uv_regularization_weights_(self):
        
        # UV position regularization weights
        try:
            del self.uv_position_weights
            self.uv_position_weights = None
        except:
            pass
        self.uv_position_weights = np.load('./models/uv_position_weights.npy') # [256,256], resized in next line
        self.uv_position_weights = cv2.resize(self.uv_position_weights, (self.framework.uv_size, self.framework.uv_size))
        with torch.no_grad():
            self.uv_position_weights = torch.from_numpy(self.uv_position_weights).to(self.device)    # convert to tensor

        # UV scale regularization weights
        try:
            del self.uv_scale_weights
            self.uv_scale_weights = None
        except:
            pass
        uv_llip_mask = cv2.imread('./models/uv_llip_mask.jpg', 0).astype(np.float32) / 255.
        uv_llip_mask = cv2.resize(uv_llip_mask, (self.framework.uv_size, self.framework.uv_size))
        uv_llip_mask = cv2.GaussianBlur(uv_llip_mask, (5, 5), 0)
        uv_scale_weights = self.framework.uv_rasterizer.uv_mask_full # [256,256], will be resized in next lines
        uv_scale_weights = uv_scale_weights - 0.5*self.framework.uv_rasterizer.uv_mask_mouth_interior
        uv_scale_weights = uv_scale_weights + 2.0*uv_llip_mask
        uv_scale_weights = cv2.resize(uv_scale_weights, (self.framework.uv_size, self.framework.uv_size))
        with torch.no_grad():
            self.uv_scale_weights = torch.from_numpy(uv_scale_weights).to(self.device)    # convert to tensor


    def _compute_average_uv_offsets_(self, personal_dataloader, batch_size=50):
        # NOTE: this function is used in personalization training only
        with torch.no_grad():
            mean_uv_offsets = None
            mean_shape_coefficients = None
            mean_shape = None
            mean_exp_coefficients = None
            videos = list(personal_dataloader.meta_data.keys())
            for video in videos:
                # randomly sample a batch
                batch_data = personal_dataloader.next_random_batch(vid=video, batch_size=batch_size)
                # flame reconstruction from coefficients
                vertices, _, _ = self.flame(shape_params=batch_data['shape'][:,...], 
                                            expression_params=batch_data['exp'][:,...], 
                                            head_pose_params=batch_data['head_pose'][:,...],
                                            jaw_pose_params=batch_data['jaw_pose'][:,...], 
                                            eye_pose_params=batch_data['eye_pose'][:,...])
                vertices = vertices.cpu().numpy()  # [N, V, 3]
                # init uv maps
                uv_maps = self.framework.create_init_uv_maps(vertices) # [N, S, S, 13]
                # uv offsets
                _, uv_offsets = self.framework.get_uv_offsets(batch_images=batch_data['img_masked'], uv_maps=uv_maps)  # [N, S, S, 13]
                # compute mena uv offsets
                temp = uv_offsets.mean(dim=0) # [S, S, 13]
                if mean_uv_offsets is None: mean_uv_offsets = temp
                else: mean_uv_offsets += temp
                if mean_shape_coefficients is None: mean_shape_coefficients = batch_data['shape'].mean(dim=0) # [100]
                else: mean_shape_coefficients += batch_data['shape'].mean(dim=0)
                if mean_shape is None: mean_shape = vertices.mean(axis=0) # [V, 3]
                else: mean_shape += vertices.mean(axis=0)
                if mean_exp_coefficients is None: mean_exp_coefficients = batch_data['exp'].mean(dim=0) # [50]
                else: mean_exp_coefficients += batch_data['exp'].mean(dim=0)
            # averge over all videos
            mean_uv_offsets /= len(videos)
            mean_shape_coefficients /= len(videos)
            mean_shape /= len(videos)
            mean_exp_coefficients /= len(videos)
            # store the data
            self.mean_uv_offsets = mean_uv_offsets[None]  # [1, S, S, 13]
            self.mean_shape = mean_shape   # [1, V, 3] np.array
            self.mean_shape_coefficients = mean_shape_coefficients[None] # [1, 100]
            self.mean_exp_coefficients = mean_exp_coefficients[None]     # [1, 50]
    
    def _render_with_global_offsets_(self, batch_data, batch_uv_delta=None):
        # NOTE: this function is used in personalization training only
        with torch.no_grad():
            # FLAME reconstruction from coefficients
            N = batch_data['exp'].shape[0]
            shape = torch.clone(self.mean_shape_coefficients)    # [1,100]
            batch_vertices = []
            for i in range(N):
                vertices, _, _ = self.flame(shape_params=shape, 
                                            expression_params=batch_data['exp'][i:i+1,...], 
                                            head_pose_params=batch_data['head_pose'][i:i+1,...],
                                            jaw_pose_params=batch_data['jaw_pose'][i:i+1,...], 
                                            eye_pose_params=batch_data['eye_pose'][i:i+1,...])
                batch_vertices.append(vertices.cpu().numpy()[0])
            batch_vertices = np.array(batch_vertices)
            # get initial UV maps from reconstructed FLAME
            uv_maps = self.framework.create_init_uv_maps(batch_vertices) # [N, S, S, 13]
            # create a copy of the mean uv offsets
            mean_uv_offsets_copy = torch.clone(self.mean_uv_offsets) # [1, S, S, 13]
        if batch_uv_delta is not None:
            # add uv delta
            uv_delta_maps = mean_uv_offsets_copy + batch_uv_delta # [N, S, S, 13]
        else:
            uv_delta_maps = mean_uv_offsets_copy # [1, S, S, 13]
        # add UV offsets to initial uv maps
        uv_maps_new = self.framework.add_uv_offsts(uv_maps, uv_delta_maps, loc_discount=0.01)
        # convert UV maps to UV Gaussians
        batch_gaussians = self.framework.uv_maps_2_gaussians(uv_maps=uv_maps_new)
        # render
        batch_rendered = self.framework.render_batch_gaussians(batch_cam_poses = batch_data['cam'], 
                                                               batch_gaussians = batch_gaussians)
        return batch_rendered
    

    def _render_with_blendmaps_(self, batch_data, uv_delta_blendmaps):
        # NOTE: this function is used in personalization training only
        # uv_delta_blendmaps: [1, S, S, 13, B]
        # Update: in v1.1, we use a learnable MLP network to map the expression coefficients to blending weights 
        """
        # v1.0
        exp = batch_data['exp'][:,:self.num_expressions] # FLAME expression coefficients [N, n_expressions]
        N = len(exp)
        blending_coefficients = exp  # [N, B]
        blending_coefficients = F.softmax(blending_coefficients, dim=1)   
        blending_coefficients = blending_coefficients.view(N, 1, 1, 1, blending_coefficients.shape[1])
        batch_uv_delta = (blending_coefficients * uv_delta_blendmaps).sum(dim=-1)  # [N, S, S, 13]
        return self._render_with_global_offsets_(batch_data, batch_uv_delta), batch_uv_delta
        """
        exp = batch_data['exp'][:,:50]      # FLAME expression coefficients [N, n_expressions]
        jaw_pose = batch_data['jaw_pose'][:,:]  # FLAME jaw pose [N, 3]
        N = len(exp)
        with torch.no_grad():
            exp_pose = torch.cat((exp, jaw_pose), dim=1)  # Concatenate exp and pose along the last dimension
            logits = self.mlp(exp_pose)
            blending_weights = F.softmax(logits, dim=1)
            _blending_weights_ = blending_weights.view(N, 1, 1, 1, blending_weights.shape[1])
        batch_uv_delta = (_blending_weights_ * uv_delta_blendmaps).sum(dim=-1)  # [N, S, S, 13]
        return self._render_with_global_offsets_(batch_data, batch_uv_delta), batch_uv_delta


    def train_global_offsets(self, personal_dataloader, batch_size=16, total_steps=300): 
        ## Stage 1

        learning_rate = 0.05
        render_size = 512
        L_uv_pos = 0.5
        L_uv_scale = 0.1e-4
        uv_delta = torch.zeros(self.mean_uv_offsets.shape, dtype=torch.float32, requires_grad=True, device=self.device) # [1, S, S, 13]
        delta_optimizer = torch.optim.Adam([uv_delta], lr=learning_rate)
        scheduler = StepLR(delta_optimizer, step_size=total_steps//20, gamma=0.9)
        lpips_loss_fn = LPIPS_LOSS_VGG(self.device)
        self.framework.set_render_size(render_size)

        # resize layer
        resize = Resize((render_size, render_size))

        # optimize
        pbar = tqdm(range(total_steps))
        for step in pbar:
            with torch.no_grad():
                # sample training data
                batch_data = personal_dataloader.next_random_batch(vid=None, batch_size=batch_size)
                targ_imgs = resize(batch_data['img_masked'].permute(0,3,1,2)) # [N,C,H,W]

            # predict the uv delta maps
            batch_uv_delta = uv_delta
            
            # render
            batch_rendered = self._render_with_global_offsets_(batch_data, batch_uv_delta=batch_uv_delta)

            # compute loss
            loss_rgb = huber_loss(batch_rendered, targ_imgs)
            loss_uv_pos_reg = compute_uv_reg_loss(uv_offsets=batch_uv_delta[:,:,:,:3], 
                                                uv_weights=self.uv_position_weights) * L_uv_pos
            loss_uv_scale_reg = compute_uv_reg_loss(uv_offsets=batch_uv_delta[:,:,:,3:6], 
                                                    uv_weights=self.uv_position_weights) * L_uv_scale
            if step > 50:
                loss_lpips = lpips_loss_fn.compute(preds=batch_rendered, targs=targ_imgs) * 0.05
            else:
                loss_lpips = 0
            loss = loss_rgb + loss_lpips + loss_uv_pos_reg + loss_uv_scale_reg

            # display status
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"Training Global...  Loss: {loss.item():.4f} LR: {current_lr:.4f}")
            
            # optimize    
            loss.backward()
            delta_optimizer.step()
            delta_optimizer.zero_grad() # clean gradient again
            scheduler.step()

        self.global_uv_delta = uv_delta


    def train_blendmaps(self, personal_dataloader, batch_size=16, total_steps=500):
        ## Stage 2

        learning_rate = 0.05
        render_size = 512
        L_uv_pos = 0.5
        L_uv_scale = 0.1e-4
        self.framework.set_render_size(render_size)

        # prepare the learnable blendmaps
        uv_delta_global = torch.clone(self.global_uv_delta)
        uv_delta_extend = uv_delta_global.unsqueeze(-1)  # [1, S, S, 13, 1]
        uv_delta_extend = uv_delta_extend.repeat(1, 1, 1, 1, self.num_expressions)  # [1, S, S, 13, B]
        uv_delta_blendmaps = nn.Parameter(uv_delta_extend)
        optimizer = torch.optim.Adam([uv_delta_blendmaps], lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=total_steps//20, gamma=0.9)
        lpips_loss_fn = LPIPS_LOSS_VGG(self.device)

        # resize layer
        resize = Resize((render_size, render_size))

        # optimize
        pbar = tqdm(range(total_steps))
        for step in pbar:
            with torch.no_grad():
                # sample training data
                batch_data = personal_dataloader.next_random_batch(vid=None, batch_size=batch_size)
                targ_imgs = resize(batch_data['img_masked'].permute(0,3,1,2)) # [N,C,H,W]
                facial_mask = torch.from_numpy(get_facial_features_mask(batch_data['parsing'])).permute(0,3,1,2).to(self.device)
            
            # render
            batch_rendered, batch_uv_delta = self._render_with_blendmaps_(batch_data, uv_delta_blendmaps)

            # compute loss
            loss_rgb = huber_loss(batch_rendered, targ_imgs)
            loss_rgb_facial = huber_loss(batch_rendered*facial_mask, targ_imgs*facial_mask) * 10
            loss_lpips = lpips_loss_fn.compute(preds=batch_rendered, targs=targ_imgs) * 0.05
            loss_uv_pos_reg = compute_uv_reg_loss(uv_offsets=batch_uv_delta[:,:,:,:3], 
                                                uv_weights=self.uv_position_weights) * L_uv_pos
            loss_uv_scale_reg = compute_uv_reg_loss(uv_offsets=batch_uv_delta[:,:,:,3:6], 
                                                    uv_weights=self.uv_position_weights) * L_uv_scale
            loss = loss_rgb + loss_rgb_facial + loss_lpips + loss_uv_pos_reg + loss_uv_scale_reg
            
            # display status
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"Training Blendmaps...  Loss: {loss.item():.4f} LR: {current_lr:.4f}")

            # optimize    
            loss.backward()
            if torch.isnan(uv_delta_blendmaps.grad).any() == False:
                optimizer.step()
            else:
                print('NaN gradient occured')

            scheduler.step()
            optimizer.zero_grad() # clean gradient again

        # store the learned blendmaps
        self.uv_delta_blendmaps = uv_delta_blendmaps


    def personal_video_training(self, personal_dataloader, batch_size=16, steps_s1=100, steps_s2=500):
        ## Initialization
        self._compute_average_uv_offsets_(personal_dataloader, batch_size=50)

        ## Stage 1: Global Rectification
        self.train_global_offsets(personal_dataloader, batch_size=batch_size, total_steps=steps_s1)

        ## Stage 2: Expression-Aware Rectification
        self.train_blendmaps(personal_dataloader, batch_size=batch_size, total_steps=steps_s2)


    def save_head_avatar(self, save_path, avatar_name):
        try:
            try: os.makedirs(os.path.join(save_path, avatar_name))
            except: print('path exists, files will be overwrite') 
            #np.save(os.path.join(save_path, avatar_name, 'mean_shape.npy'), self.mean_shape)
            torch.save(self.mean_uv_offsets.cpu(), os.path.join(save_path, avatar_name, 'mean_uv_offsets.pt'))
            torch.save(self.mean_shape_coefficients.cpu(), os.path.join(save_path, avatar_name, 'mean_shape_coefficients.pt'))
            torch.save(self.mean_exp_coefficients.cpu(), os.path.join(save_path, avatar_name, 'mean_exp_coefficients.pt'))
            torch.save(self.global_uv_delta.cpu(), os.path.join(save_path, avatar_name, 'global_uv_delta.pt'))
            torch.save(self.uv_delta_blendmaps.cpu(), os.path.join(save_path, avatar_name, 'uv_delta_blendmaps.pt'))
            print(f'Head avatar parameters saved to {os.path.join(save_path, avatar_name)}')
        except Exception as e:
            print('dejavu.py function save_head_avatar() : ', e)


    def load_head_avatar(self, save_path, avatar_name):
        try:
            for cache in [self.mean_uv_offsets, self.mean_shape_coefficients, self.mean_shape,
                          self.mean_exp_coefficients, self.global_uv_delta, self.uv_delta_blendmaps]:
                del cache
                cache = None
            #self.mean_shape = np.load(os.path.join(save_path, avatar_name, 'mean_shape.npy'))
            self.mean_uv_offsets = torch.load(os.path.join(save_path, avatar_name, 'mean_uv_offsets.pt')).to(self.device)
            self.mean_shape_coefficients = torch.load(os.path.join(save_path, avatar_name, 'mean_shape_coefficients.pt')).to(self.device)
            self.mean_exp_coefficients = torch.load(os.path.join(save_path, avatar_name, 'mean_exp_coefficients.pt')).to(self.device)
            self.global_uv_delta = torch.load(os.path.join(save_path, avatar_name, 'global_uv_delta.pt')).to(self.device)
            self.uv_delta_blendmaps = torch.load(os.path.join(save_path, avatar_name, 'uv_delta_blendmaps.pt')).to(self.device)
            # check UV gaussian map size
            uv_map_size = self.mean_uv_offsets.shape[1]
            if uv_map_size != self.framework.uv_size:
                self.framework.set_uv_size(uv_map_size)
                self._init_uv_regularization_weights_()
                print(f'Size of UV Gaussian map changed to {uv_map_size}x{uv_map_size}')
                print('Number of Gaussians: ', len(self.framework.uv_rasterizer.valid_coords[0]))
            # check number of expressions (used in UV blending)
            num_expressions = self.uv_delta_blendmaps.shape[-1]
            if num_expressions != self.num_expressions:
                self.num_expressions = num_expressions
                print(f'Number of expression coefficients changed to {num_expressions}.')
            print('Head avatar parameters loaded')
        except Exception as e:
            print('dejavu.py function load_head_avatar() : ', e)


    @torch.no_grad()
    def drive_head_avatar(self, exp:np.array = None, 
                                head_pose:np.array = None,
                                jaw_pose:np.array = None, 
                                eye_pose:np.array = None, 
                                cam_pose:np.array = None, 
                                return_all = False, return_gaussians = False, exp_alpha=0.9):
        """
        Use FLAME coefficients and camera pose to drive the Gaussian Dejavu head avatar

        inputs:
            -      exp: FLAME expression coefficients [1, 50]
            - head_pose: FLAME head pose [1, 3]
            - jaw_pose: FLAME jaw pose [1, 3]
            - eye_pose: FLAME eyeballs poses [1, 6]
            - cam_pose: camera pose [1, 6]   rotation + translation --> [yaw, pitch, roll, dx, dy, dz]
            - return_all: return all rendered images or not
            - return_gaussians: return 3D Gaussians or not
            - exp_alpha: expression blending factor
        output:
            - batch_rendered
            - batch_gaussians (optional)
        """
        N = 1

        if exp is None:
            exp = np.zeros([1,50], dtype=np.float32)
        if head_pose is None:
            head_pose = np.zeros([1,3], dtype=np.float32)
        if jaw_pose is None:
            jaw_pose = np.zeros([1,3], dtype=np.float32)
        if eye_pose is None:
            eye_pose = np.zeros([1,6], dtype=np.float32)
        if cam_pose is None:
            cam_pose = np.array([[0,0,0,0,0,1.0]], dtype=np.float32)

        # convert to Pytorch tensors
        shape = torch.clone(self.mean_shape_coefficients)         # [1,100]
        exp_base = torch.clone(self.mean_exp_coefficients)        # [1,50]
        exp_drive = torch.from_numpy(exp).to(self.device)         # [1,50]
        exp = exp_drive # (1-exp_alpha)*exp_base + exp_alpha*exp_drive
        head_pose = torch.from_numpy(head_pose).to(self.device)   # [1,3]
        jaw_pose = torch.from_numpy(jaw_pose).to(self.device)     # [1,3]
        eye_pose = torch.from_numpy(eye_pose).to(self.device)     # [1,6]
        cam_pose = torch.from_numpy(cam_pose).to(self.device)     # [1,6]

        # FLAME neutral shape reconstruction
        vertices, _, _ = self.flame(shape_params=shape, expression_params=exp, 
                                    head_pose_params=head_pose, jaw_pose_params=jaw_pose, eye_pose_params=eye_pose)

        # FLAME shape rasterization to UV map
        uv_maps = self.framework.create_init_uv_maps(vertices.cpu().numpy())   # [N, S, S, 13]

        # compute blending weights
        # blending_coefficients = F.softmax(exp[:,:self.num_expressions], dim=1) # v1.0
        exp_pose = torch.cat((exp, jaw_pose[:,:]), dim=1)  # Concatenate exp and jaw pose along the last dimension
        logits = self.mlp(exp_pose)
        blending_weights = F.softmax(logits, dim=1)

        # expression-aware uv blending
        blending_weights = blending_weights.view(N, 1, 1, 1, blending_weights.shape[1])
        batch_uv_delta = (blending_weights * self.uv_delta_blendmaps).sum(dim=-1)  # [N, S, S, 13]        
        mean_uv_offsets_copy = torch.clone(self.mean_uv_offsets) # [1, S, S, 13]
        uv_delta_maps = mean_uv_offsets_copy + batch_uv_delta    # [N, S, S, 13]

        # add UV offsets to initial uv maps
        uv_maps_new = self.framework.add_uv_offsts(uv_maps, uv_delta_maps, loc_discount=0.01)
        
        # convert UV Gaussian maps to UV Gaussians that support by 3DGS renderer
        batch_gaussians = self.framework.uv_maps_2_gaussians(uv_maps=uv_maps_new)

        # render
        batch_rendered = self.framework.render_batch_gaussians(batch_cam_poses = cam_pose, 
                                                               batch_gaussians = batch_gaussians,
                                                               return_all = return_all)
        
        if return_gaussians:
            return batch_rendered, batch_gaussians
        else:
            return batch_rendered
    
