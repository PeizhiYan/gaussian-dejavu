######################################
## Utility functions for Personal    #
## Video Data.                       #
## Author: Peizhi Yan                #
##   Date: 07/19/2024                #
## Update: 02/28/2025                #
######################################

import numpy as np
#import open3d as o3d
#import copy
#import os
#import matplotlib.pyplot as plt
import cv2
import PIL
import scipy
import torch
import pickle
from time import time
import os
import random



def mask_out_background(imgs : np.array, parsing : np.array, return_masks = False):
    """
    Given image as well as its parsing mask,
    mask out (set as white) the background.
    inputs:
        - img: [N, H, W, C]        np.uint8  0 ~ 255
        - parsing: [N, 512, 512]   np.uint8
    returns:
        - imgs_masked: [N, 512, 512, C]        np.float32  0 ~ 1.0
        - mask (optional): [N, 512, 512, 1] 
    """
    # Resize the images
    resized_imgs = [cv2.resize(img, (512, 512)) for img in imgs]
    resized_imgs = np.array(resized_imgs, dtype=np.float32)

    # Convert images to float32 for processing
    imgs_float = resized_imgs / 255.0

    # Expand parsing mask dimensions to match imgs for broadcasting
    parsing_expanded = np.expand_dims(parsing, -1)
    
    # Create a mask where parsing == 0 (background), then invert it to target the foreground
    mask = parsing_expanded != 0
    cloth_mask = parsing_expanded == 16
    mask = mask ^ cloth_mask
    neck_mask = (parsing_expanded == 14) | (parsing_expanded == 15)
    mask = mask ^ neck_mask
    
    # Apply mask, setting background to white (1.0)
    imgs_masked = np.where(mask, imgs_float, 1.0)

    if return_masks:
        return imgs_masked, mask
    else:
        return imgs_masked

def get_face_mask(parsing : np.array):
    """
    Given parsing mask get the face region mask.
    {
     0: 'background'
     1: 'skin', 
     2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 6: 'eye_g', 
     7: 'l_ear', 8: 'r_ear', 9: 'ear_r', 
     10: 'nose', 
     11: 'mouth', 12: 'u_lip', 13: 'l_lip', 
     14: 'neck', 15: 'neck_l', 
     16: 'cloth', 17: 'hair', 18: 'hat'
    }
    inputs:
        - parsing: [N, 512, 512]   np.uint8
    returns:
        - face_mask: [N, 512, 512, 1] 
    """
    # Expand parsing mask dimensions to match imgs for broadcasting
    parsing_expanded = np.expand_dims(parsing, -1)
    
    # Create a mask where parsing == 0 (background), then invert it to target the foreground
    face_mask = parsing_expanded != 0 # remove background first
    non_face_mask = parsing_expanded >= 14 # non-facial region
    ear_mask = (parsing_expanded >= 7) & (parsing_expanded <=9) # ears
    face_mask = (face_mask ^ non_face_mask) ^ ear_mask
    
    return face_mask

def get_facial_features_mask(parsing : np.array):
    """
    Given parsing mask get the facial features region mask:
    Eyebrows, Eyes, Nose, Mouth (including lips).
    {
     0: 'background'
     1: 'skin', 
     2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 6: 'eye_g', 
     7: 'l_ear', 8: 'r_ear', 9: 'ear_r', 
     10: 'nose', 
     11: 'mouth', 12: 'u_lip', 13: 'l_lip', 
     14: 'neck', 15: 'neck_l', 
     16: 'cloth', 17: 'hair', 18: 'hat'
    }
    inputs:
        - parsing: [N, 512, 512]   np.uint8
    returns:
        - features_mask: [N, 512, 512, 1] 
    """
    # Expand parsing mask dimensions to match imgs for broadcasting
    parsing_expanded = np.expand_dims(parsing, -1)
    
    mask_1 = (parsing_expanded >= 2) & (parsing_expanded <=5)   # eyes and eyebrows
    mask_2 = (parsing_expanded >= 10) & (parsing_expanded <=13) # nose and mouth
    
    features_mask = mask_1 | mask_2

    return features_mask


def convert_mov_to_images(video_path, original_fps = 60, subsample_fps = 30):
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # Calculate the subsampling interval
    interval = original_fps / subsample_fps
    
    # Select every nth frame based on the interval
    subsampled_frames = [frames[int(i * interval)] for i in range(int(len(frames) / interval))]

    frame_count = len(subsampled_frames)
    
    cap.release()
    print(f"Conversion completed. {frame_count}")

    return subsampled_frames




#############################
##       Data Loader       ##
#############################
class PersonalDataLoader():    
    def __init__(self, dataset_path: str,
                 shuffle : bool = True, device : str = 'cpu'):
        """
        Inputs:
            - dataset_path (str): Path to the dataset directory.
            - shuffle (bool): If True, data will be shuffled at each epoch.
            - device (str): "cpu" or CUDA device such as "cuda:0"
        """
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.device = device
        self.subject_name = os.path.basename(dataset_path)

        self.meta_data = {} # example usage: file_path = meta_data[vid][fid]
            
        # loop through video ids
        path_a = self.dataset_path
        for vid in os.listdir(path_a):
            if vid.startswith('.'):
                continue
            else:
                self.meta_data[vid] = {}
            
            # loop through frame ids
            path_b = os.path.join(path_a, vid)
            for fid in os.listdir(path_b):
                if fid.startswith('.'):
                    continue
                else:
                    if fid.endswith('.npy') and not fid.startswith('.'):
                        self.meta_data[vid][fid.split('.')[0]] = os.path.join(path_b, fid)

        self.n_frames = 0
        for vid in self.meta_data.keys():
            for fid in self.meta_data[vid].keys():
                self.n_frames += 1
                    
        print(f'Number of frames: {self.n_frames}.')

    def batch_from_frame_ids(self, vid, frame_ids=[]):
        # given video id, randomly sample a batch of frames
        
        if vid not in self.meta_data.keys():
            return None
        
        real_batch_size = len(frame_ids)
        if real_batch_size == 0:
            return None

        batch_data = {
            'vid': vid, # video id
            'fid': [],  # frame ids
            'img': np.zeros([real_batch_size, 512, 512, 3], dtype=np.uint8),
            'parsing':      np.zeros([real_batch_size, 512, 512], dtype=np.uint8),
            'vertices': np.zeros([real_batch_size, 5023, 3], dtype=np.float32),
            'shape':    np.zeros([real_batch_size, 100], dtype=np.float32),
            'exp':      np.zeros([real_batch_size, 50], dtype=np.float32),
            'pose':     np.zeros([real_batch_size, 6], dtype=np.float32),
            'eye_pose': np.zeros([real_batch_size, 6], dtype=np.float32),
            'tex':      np.zeros([real_batch_size, 50], dtype=np.float32),
            'light':    np.zeros([real_batch_size, 9, 3], dtype=np.float32),
            'cam':      np.zeros([real_batch_size, 6], dtype=np.float32),
        }
        
        # load batch data
        for i, fid in enumerate(frame_ids):
            file_path = self.meta_data[vid][fid]
            # load data
            loaded = np.load(file_path, allow_pickle=True)
            # store data
            batch_data['fid'].append(fid)
            batch_data['vertices'][i] = loaded['vertices']
            batch_data['img'][i] = loaded['img']
            batch_data['parsing'][i] = loaded['parsing'].astype(np.uint8)
            batch_data['shape'][i] = loaded['shape'][0]
            batch_data['exp'][i] = loaded['exp'][0,:50] # we only use the first 50 expression coefficients
            batch_data['pose'][i] = loaded['pose'][0]
            batch_data['eye_pose'][i] = loaded['eye_pose'][0]
            batch_data['tex'][i] = loaded['tex'][0]
            batch_data['light'][i] = loaded['light'][0]
            batch_data['cam'][i] = loaded['cam']


        # mask out background
        batch_data['img_masked'], batch_data['masks'] = mask_out_background(
                                                        imgs=batch_data['img'], 
                                                        parsing=batch_data['parsing'],
                                                        return_masks=True)
        # convert to tensor
        for key in batch_data:
            if key not in ['vid', 'fid', 'vertices', 'parsing']:
                batch_data[key] = torch.from_numpy(batch_data[key]).detach().to(self.device)
            
        return batch_data

    def batch_from_range(self, vid, fid_start=0, fid_end=None):
        # given video id, return batch of frames in the given range

        # collect the frame ids
        frame_ids = np.array(sorted(list(self.meta_data[vid].keys()),key=int))
        n_frames = len(frame_ids)

        if fid_end != None:
            if fid_start > fid_end or fid_end > n_frames:
                return None
        else:
            fid_end = n_frames
                
        frame_ids = frame_ids[fid_start:fid_end+1] # include the end id
        return self.batch_from_frame_ids(vid, frame_ids)

    def next_random_batch(self, vid=None, batch_size=16):
        if vid is None:
            # randomly pick a video
            vid = random.choice(list(self.meta_data.keys()))

        # collect the frame ids
        frame_ids = np.array(sorted(list(self.meta_data[vid].keys()),key=int))
        n_frames = int(len(frame_ids) * 0.9)
        frame_ids = frame_ids[:n_frames] # take the first 90% frames for training

        p = np.random.permutation(n_frames)
        frame_ids = frame_ids[p] # random shuffle
        
        frame_ids = frame_ids[:batch_size]
        real_batch_size = len(frame_ids)
        
        return self.batch_from_frame_ids(vid, frame_ids)



