######################################
## FLAME mesh to UV map rasterizer.  #
## Author: Peizhi Yan                #
##   Date: 03/18/2024                #
## Update: 10/31/2024                #
######################################

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.interpolate import interp1d

## Pytorch3D
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes

## Utilities 
from utils.general_utils import dict2obj





class Pytorch3dRasterizer(nn.Module):
    """
    Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=256):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin':  None,
            'perspective_correct': False,
        }
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings.image_size
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
            
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone() 
        attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)

        return pixel_vals




def generate_face_vertices(vertices, faces):
    """ 
    # borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def generate_triangles(h, w, margin_x=2, margin_y=5, mask = None):
    # borrowed from: https://github.com/yfeng95/DECA/blob/master/decalib/utils/util.py
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    #.
    # w*h
    triangles = []
    for x in range(margin_x, w-1-margin_x):
        for y in range(margin_y, h-1-margin_y):
            triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
            triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:,[0,2,1]]
    return triangles


def separate_eyes(uv_mask):
    # Author: Peizhi Yan
    # To separate the eyes from the original UV mask
    # - uv_mask: [S, S]
    uv_mask_size = uv_mask.shape[0]
    uv_mask_full = np.array(uv_mask)
    uv_mask_no_eyes = np.array(uv_mask_full)
    
    # remove left-side eye
    y_indices = np.arange(uv_mask_no_eyes.shape[1])
    x_line = -1 * y_indices + int((61/256)*uv_mask_size)
    x_indices = np.arange(uv_mask_no_eyes.shape[0])[:, None]
    uv_mask_no_eyes = np.where(x_indices > x_line, uv_mask_no_eyes, 0)
    
    # remove right-side eye
    y_indices = np.arange(uv_mask_no_eyes.shape[1])
    x_line = 1 * y_indices - int((194/256)*uv_mask_size)
    x_indices = np.arange(uv_mask_no_eyes.shape[0])[:, None]
    uv_mask_no_eyes = np.where(x_indices > x_line, uv_mask_no_eyes, 0)

    uv_mask_eyes = uv_mask - uv_mask_no_eyes

    return uv_mask_full, uv_mask_no_eyes, uv_mask_eyes


def interpolate_2d_array(data, M, kind='linear'):
    """
    Interpolate a 2D numpy array with shape [N, 3] to shape [M, 3].
    
    Parameters:
        data (numpy.ndarray): Original data array of shape [N, 3].
        M (int): Number of points in the interpolated array.
        kind (str): Type of interpolation (e.g., 'linear', 'cubic').
    
    Returns:
        numpy.ndarray: Interpolated data array of shape [M, 3].
    """
    if data.shape[1] != 3:
        raise ValueError("Input array must have shape [N, 3]")
    
    N = data.shape[0]
    x = np.linspace(0, N-1, N)  # Original x values
    x_new = np.linspace(0, N-1, M)  # New x values for interpolation
    
    # Interpolating each dimension independently
    f0 = interp1d(x, data[:, 0], kind=kind)
    f1 = interp1d(x, data[:, 1], kind=kind)
    f2 = interp1d(x, data[:, 2], kind=kind)
    
    # Getting the interpolated values
    data_new = np.zeros((M, 3))
    data_new[:, 0] = f0(x_new)
    data_new[:, 1] = f1(x_new)
    data_new[:, 2] = f2(x_new)
    
    return data_new


def interpolate_between_arrays(A, B, K):
    """
    Interpolate between two arrays A and B to create a new array C with shape [K, M, 3].
    
    Parameters:
        A (numpy.ndarray): Starting array with shape [M, 3].
        B (numpy.ndarray): Ending array with shape [M, 3].
        K (int): Number of interpolated steps.
    
    Returns:
        numpy.ndarray: Interpolated array with shape [K, M, 3].
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    if A.shape[1] != 3:
        raise ValueError("A and B must have shape [M, 3]")
    
    M = A.shape[0]
    # Initialize the array C
    C = np.zeros((K, M, 3))
    
    # Create interpolation ratios
    for i in range(K):
        ratio = i / (K - 1)
        C[i] = A * (1 - ratio) + B * ratio
    
    return C


# predefined indices of the mouth interior lines on FLAME mesh
mouth_line_upper_indices = [1572, 1594, 1595, 1746, 1747, 1742, 1739, 1665, 1666, 3514, 2783, 2782, 2854, 2857, 2862, 2861, 2731, 2730, 2708]
mouth_line_lower_indices = [1572, 1573, 1860, 1862, 1830, 1835, 1852, 3497, 2941, 2933, 2930, 2945, 2943, 2709, 2708]
mouth_line_end_indices = [1572, 2708]
def generate_mouth_interior_uv(vertices : np.array, M : int, K : int):
    """
    Generate the UV position map for the mouth interior

    Parameters:
        vertices: [V, 3], np.array
        M: The width of UV mouth interior patch
        K: The half-height of UV mouth interior patch 

    Returns:
        mouth_interior_uv_patch: [2K, M, 3], np.array
    """
    # retrieve mouth lines vertices
    upper = np.copy(vertices[mouth_line_upper_indices])
    lower = np.copy(vertices[mouth_line_lower_indices])
    endline = np.copy(vertices[mouth_line_end_indices])
    endline[:,2] = endline[:,2] - 0.02 # move the endline to the back
    
    # interpolate mouth lines
    upper_interpolated = interpolate_2d_array(data=upper, M=M) # [M, 3]
    lower_interpolated = interpolate_2d_array(data=lower, M=M) # [M, 3]
    endline_interpolated = interpolate_2d_array(data=endline, M=M) # [M, 3]
        
    # interpolate the mouth interior surfaces vertices
    upper_surface_interpolated = interpolate_between_arrays(A=upper_interpolated, B=endline_interpolated, K=K) # [K, M, 3]
    lower_surface_interpolated = interpolate_between_arrays(A=endline_interpolated, B=lower_interpolated, K=K) # [K, M, 3]
    
    # covnert the mouth interior vertices to UV patch
    mouth_interior_uv_patch = np.zeros([K*2, M, 3], dtype=np.float32)
    mouth_interior_uv_patch[:K, :, :] = upper_surface_interpolated
    mouth_interior_uv_patch[K:, :, :] = lower_surface_interpolated

    return mouth_interior_uv_patch






class UV_Rasterizer(nn.Module):
    def __init__(self, obj_path, uv_size=256, device='cpu'):
        """
        This class is not designed to be differentiable currently
        inputs:
            - obj_path: the path of FLAME head mesh template .obj
            - uv_size:  the rasterized UV map size
        returns:
            - rasterized UV map: np.array [S,S,3]
        """
        super(UV_Rasterizer, self).__init__()
        self.device = device
        self.uv_size = uv_size
        self.uv_rasterizer = Pytorch3dRasterizer(uv_size).to(self.device)
        verts, faces, aux = load_obj(obj_path)
        uvcoords = aux.verts_uvs[None, ...].to(self.device)      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...].to(self.device)  # (N, F, 3)
        faces = faces.verts_idx[None,...].to(self.device)
        n_verts = uvcoords.shape[1]
        
        # faces
        dense_triangles = generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None,:,:].to(self.device))
        self.register_buffer('faces', faces.to(self.device))
        self.register_buffer('raw_uvcoords', uvcoords.to(self.device))

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1).to(self.device) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = generate_face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords.to(self.device))
        self.register_buffer('uvfaces', uvfaces.to(self.device))
        self.register_buffer('face_uvcoords', face_uvcoords.to(self.device))

        # uv mask: indicate the valid UV pixels
        self.uv_mask = np.zeros([uv_size, uv_size], dtype=np.uint8)
        temp = self.rasterize(vertices=np.ones([n_verts, 3], dtype=np.float32), mouth_interior=False)
        neck_cut_line = int((200/256)*uv_size)
        temp[neck_cut_line:,:] *= 0 # remove the neck
        self.valid_coords = np.where(temp > 0)[:2]
        self.uv_mask[self.valid_coords] = 1.0

        # add mouth interior to the UV map
        self.M = 50
        self.K = ((uv_size - neck_cut_line) - 10) // 2 # we leave 5 pixels top and bottom
        self.m_i = neck_cut_line + 5
        self.m_j = uv_size // 2 - self.M // 2
        self.uv_mask_mouth_interior = np.zeros([uv_size, uv_size], dtype=np.uint8)
        self.uv_mask_mouth_interior[self.m_i : self.m_i + 2*self.K, self.m_j : self.m_j + self.M] = 1.0

        # generate three sets of uv masks
        self.uv_mask = self.uv_mask + self.uv_mask_mouth_interior
        self.uv_mask_full, self.uv_mask_no_eyes, self.uv_mask_eyes = separate_eyes(self.uv_mask)
        self.valid_coords = np.where(self.uv_mask_full > 0)[:2] # update the valid coordinates

    
    def rasterize(self, vertices, mouth_interior=False):
        '''
        borrowed from: https://github.com/yfeng95/DECA/blob/master/decalib/utils/renderer.py
        warp vertices from world space to uv space
        vertices: [V, 3], np.array
        uv_vertices: [h, w, 3] np.array
        '''
        with torch.no_grad():
            vertices_np = vertices
            vertices = torch.from_numpy(vertices)[None].to(self.device) # [1, V, 3]
            batch_size = vertices.shape[0]
            face_vertices = generate_face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
            uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1), 
                                             self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
            uv_vertices = uv_vertices[0].cpu().permute(1,2,0).numpy() # [h, w, 3]
            if mouth_interior:
                mouth_interior_uv_patch = generate_mouth_interior_uv(vertices=vertices_np, M=self.M, K=self.K)
                uv_vertices[self.m_i : self.m_i + 2*self.K, self.m_j : self.m_j + self.M, :] = mouth_interior_uv_patch
        
        return uv_vertices


