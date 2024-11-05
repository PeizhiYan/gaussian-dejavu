######################################
## Loss Functions.                   #
## Author: Peizhi Yan                #
##   Date: 03/28/2024                #
## Update: 05/28/2024                #
######################################

import torch
import torch.nn.functional as F
from lpips import LPIPS # lpips loss
from utils.personal_video_utils import get_facial_features_mask, get_face_mask
import numpy as np
import pytorch_msssim   # ssim loss, source https://github.com/VainF/pytorch-msssim




def get_facial_feature_weights(parsing : np.array, device):
    """
        - parsing:   [N, H, W]
    returns:
        - facial_feature_weights:    [N, 1, H, W]
    """
    ffm = get_facial_features_mask(parsing)  # [N,H,W,1]
    facial_feature_weights = np.ones(ffm.shape, dtype=np.float32)
    facial_feature_weights = facial_feature_weights + 10 * ffm.astype(np.float32)
    return torch.from_numpy(facial_feature_weights).permute(0,3,1,2).to(device)

def get_facial_weights(parsing : np.array, gain : int, device):
    """
        - parsing:   [N, H, W]
    returns:
        - facial_weights:    [N, 1, H, W]
    """
    face_mask = get_face_mask(parsing)  # [N,H,W,1]
    facial_weights = np.ones(face_mask.shape, dtype=np.float32)
    facial_weights = facial_weights + gain * face_mask.astype(np.float32)
    return torch.from_numpy(facial_weights).permute(0,3,1,2).to(device)


def compute_l1_loss(pred_imgs : torch.tensor, targ_imgs : torch.tensor, 
                    weights : torch.tensor = None):
    """
        - pred_imgs: [N, C, H, W]   0 ~ 1.0 
        - targ_imgs: [N, C, H, W]   0 ~ 1.0
        - weights (optional):   [N, 1, H, W]
    """
    if weights is None:
        return F.l1_loss(pred_imgs, targ_imgs, reduction='mean')
    else:
        l1_loss = F.l1_loss(pred_imgs, targ_imgs, reduction='none') # [N,C,H,W]
        l1_loss = l1_loss * weights
        return torch.mean(l1_loss)


def compute_face_reg_loss(uv_position_offsets : torch.tensor,
                          uv_face_weights : torch.tensor):
    """
    NOTE: this function will be deprecated after V1.1
          replaced by compute_uv_reg_loss()
    - uv_position_offsets: [N,256,256,3]
    - uv_face_weights: [256,256]
    """
    l2 = torch.norm(uv_position_offsets, p=2, dim=-1) # [N,256,256]
    l2 = l2 * uv_face_weights # [N,256,256] element-wise multiplication (broadcasted across the batch)
    return torch.mean(l2)


def compute_uv_reg_loss(uv_offsets : torch.tensor,
                        uv_weights : torch.tensor):
    """
    - uv_offsets: [N,256,256,C]
    - uv_weights: [256,256]
    """
    l2 = torch.norm(uv_offsets, p=2, dim=-1) # [N,256,256]
    l2 = l2 * uv_weights # [N,256,256] element-wise multiplication (broadcasted across the batch)
    return torch.mean(l2)


class LPIPS_LOSS_VGG():
    def __init__(self, device) -> None:
        # image should be RGB, IMPORTANT: normalized to [-1,1]
        self.lpips_loss_fn_vgg = LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

    def compute(self, preds, targs):
        """
        input size NCHW
        input RGB range [0, 1]
        """
        loss = self.lpips_loss_fn_vgg(preds*2-1, targs*2-1)
        loss = torch.mean(loss) # average on batch
        return loss


def compute_view_consistency_loss(uv_offsets : torch.tensor, channels = None):
    """
    - uv_offsets: [N,256,256,C]
    - channels: the list of channels to be used in loss computation, 
                default is None, all channels are used
    """
    if channels is None:
        mean_uv_offsets = torch.mean(uv_offsets, dim=0) # [H,W,C]
        diff = uv_offsets - mean_uv_offsets # [N,H,W,C]
    else:
        mean_uv_offsets = torch.mean(uv_offsets[:,:,:,channels], dim=0) # [H,W,C]
        diff = uv_offsets[:,:,:,channels] - mean_uv_offsets # [N,H,W,C]
    loss = torch.norm(diff, p=2, dim=-1) # [N,H,W]
    loss = torch.mean(loss)
    return loss


def compute_canonical_fmap_loss(fmaps):
    """
    - fmaps: [N,Sf,Sf,F]
    """
    mean_fmaps = torch.mean(fmaps, dim=0)  #  [Sf,Sf,F]
    diff = fmaps - mean_fmaps              #  [N,Sf,Sf,F]
    loss = torch.norm(diff, p=2, dim=-1)   #  [N,Sf,Sf]
    loss = torch.mean(loss)
    return loss

ssim_loss_fn = pytorch_msssim.MS_SSIM(data_range=1.0)


def huber_loss(network_output, gt, alpha=0.1):
    diff = torch.abs(network_output - gt)
    mask = (diff < alpha).float()
    loss = 0.5*diff**2*mask + alpha*(diff-0.5*alpha)*(1.-mask)
    return loss.mean()







