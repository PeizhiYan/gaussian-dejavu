#
# Gaussian DejaVu Head Avatar Driver Demo
# 
# Copyright 2024. Peizhi Yan
#


## Enviroment Setup
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set the visible CUDA, here we use the second GPU
# WORKING_DIR = '/home/peizhi/Documents/gaussian-dejavu/'
# os.chdir(WORKING_DIR) # change the working directory to the project's absolute path
# sys.path.append(WORKING_DIR)
print("Current Working Directory: ", os.getcwd())


sys.path.append('./models')
sys.path.append('./networks')
sys.path.append('./utils')
sys.path.append('./utils/flame_lib/')
sys.path.append('./utils/diff-gaussian-rasterization')
sys.path.append('./utils/gaussian_renderer')
sys.path.append('./utils/scene')
sys.path.append('./utils/arguments')
sys.path.append('./utils/simple-knn')

import dearpygui.dearpygui as dpg
from tkinter import Tk, filedialog
from time import time
from time import sleep
from tqdm import tqdm
import numpy as np
from PIL import Image
import math
import cv2
import threading

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dejavu import GaussianDejavu
from utils.personal_video_utils import *

from models.mediapipe_to_flame import MP_2_FLAME, compute_head_pose_from_mp_landmarks_3d
from utils.mp2dlib import convert_landmarks_mediapipe_to_dlib
from utils.image_utils import image_align


def min_max_normalize(image):
    norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return norm_image


def compute_camera_offsets(yaw, pitch, radius):
    # orbit camera function
    dx = math.sin(yaw) * radius * 0.9       # * 0.9 to hide part of the back of head
    dy = - math.sin(pitch) * radius * 0.9   # * 0.9 to hide part of the back of head
    dz_a = radius - math.cos(yaw) * radius
    dz_b = radius - math.cos(pitch) * radius
    dz = dz_a + dz_b
    return dx, dy, dz


def blur_head_boundary(rendered_img, blur_kernel_size=25, erode_kernel_size=20, sigma=5):
    # rendered_img: RGB numpy array, float32
    # Ensure image is in 0-255 range if given in 0-1
    rendered_img = (np.clip(rendered_img, 0, 1.0) * 255).astype(np.uint8)

    # Create binary mask for head
    gray = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    # Erode mask to define boundary area
    eroded_mask = 255 - cv2.erode(binary_mask, np.ones((erode_kernel_size, erode_kernel_size), np.uint8), iterations=1)

    # Blur the boundary area
    blurred_boundary = cv2.GaussianBlur(rendered_img, (blur_kernel_size, blur_kernel_size), sigmaX=sigma, sigmaY=sigma)
    rendered_img = cv2.GaussianBlur(rendered_img, (5,5), sigmaX=1, sigmaY=1)

    # Create alpha mask for blending
    alpha = cv2.GaussianBlur(eroded_mask.astype(float) / 255.0, (blur_kernel_size, blur_kernel_size), sigmaX=sigma*2)

    # Blend the original image with blurred boundary
    blurred_img = (alpha[..., None] * blurred_boundary + (1 - alpha[..., None]) * rendered_img)

    return blurred_img / 255.


def mediapipe_face_detection(mediapipe_detector, img):
    """
    Run Mediapipe face detector
    input:
        - img: image data  numpy uint8
    output:
        - lmks_dense: landmarks numpy [478, 3], the locations are normalized
        - blend_scores: facial blendshapes numpy [52]
    """
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img) # convert numpy image to Mediapipe Image

    # Detect face landmarks from the input image.
    detection_result = mediapipe_detector.detect(image)

    if len(detection_result.face_blendshapes) == 0:
        return None, None

    # Post-process mediapipe face blendshape scores
    blend_scores = detection_result.face_blendshapes[0]
    blend_scores = np.array(list(map(lambda l: l.score, blend_scores)), dtype=np.float32)

    # Post-process mediapipe dense face landmarks, re-scale to image space 
    lmks_dense = detection_result.face_landmarks[0] # the first detected face
    lmks_dense = np.array(list(map(lambda l: np.array([l.x, l.y, l.z]), lmks_dense)))
    lmks_dense[:, 0] = lmks_dense[:, 0] * img.shape[1]
    lmks_dense[:, 1] = lmks_dense[:, 1] * img.shape[0]

    return lmks_dense, blend_scores





"""
Load Mediapipe face detector
"""
base_options = python.BaseOptions(model_asset_path='./models/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=False,
                                        num_faces=1)
mediapipe_detector = vision.FaceLandmarker.create_from_options(options)



"""
Mediapipe to FLAME mapping module
"""
mp2flame = MP_2_FLAME(mappings_path='./models/mediapipe_to_flame/mappings')



"""
Initialize webcam capture
"""
try:
    cap = cv2.VideoCapture(0)
    print('Webcam started.')
except:
    print('Webcam cannot start. Please check.')
    exit()



"""
Create Gaussian Dejavu Pipeline
"""
dejavu = GaussianDejavu(network_weights='./models/dejavu_network.pt', uv_map_size=120)
device = dejavu.device










"""
GUI Parameters
"""
WINDOW_H = 700
WINDOW_W = 1024 + 300
RENDER_SIZE = dejavu.framework.H
DISPLAY_Y = (WINDOW_H - RENDER_SIZE) // 2
DISPLAY_X = (WINDOW_W - RENDER_SIZE) // 2
WEBCAM_H = 256
WEBCAM_W = 256


"""
Global Variables
"""
avatar_loaded = True ; dejavu.load_head_avatar(save_path='./saved_avatars', avatar_name='peizhi-uv320-1.1')
#avatar_loaded = False
display_buffer = np.ones((WINDOW_H, WINDOW_W, 3), dtype=np.float32)
webcam_display_buffer = np.zeros((WEBCAM_H, WEBCAM_W, 3), dtype=np.float32)
last_time = time()  # Initialize the last time for FPS calculation
last_fps = 0
to_blur = True
prev_x, prev_y = None, None
is_dragging_in_canvas = False
last_update_time = time()
is_webcam_running = True
is_webcam_update = True
exp = np.zeros([1,50], dtype=np.float32)     # FLAME parameters
pose = np.zeros([1,6], dtype=np.float32)     # FLAME parameters
eye_pose = np.zeros([1,6], dtype=np.float32) # FLAME parameters
yaw_realign, pitch_realign = 0, 0
prev_blendshape_scores = np.zeros([52], dtype=np.float32)


def update_webcam_frame():
    global is_webcam_running, is_webcam_update, webcam_display_buffer
    global exp, pose, eye_pose
    global yaw_realign, pitch_realign
    global prev_blendshape_scores

    while is_webcam_running:
        while is_webcam_update:
            ret, frame = cap.read()
            if ret:
                # Convert the frame to RGB and scale to [0, 1]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
                img_h = frame_rgb.shape[0]
                img_w = frame_rgb.shape[1]

                # Run Mediapipe face detector
                lmks_dense, blendshape_scores = mediapipe_face_detection(mediapipe_detector, img=frame_rgb)
                if lmks_dense is None: continue

                # Smoothing blendshape scores
                SMOOTHING_FACTOR = 0.8
                blendshape_scores = SMOOTHING_FACTOR*blendshape_scores + (1-SMOOTHING_FACTOR)*prev_blendshape_scores
                blendshape_scores = np.clip(0,1.0, blendshape_scores)
                prev_blendshape_scores = blendshape_scores

                # Get FLAME driving signal from detected Mediapipe blendshape scores
                exp, pose, eye_pose = mp2flame.convert(blendshape_scores=blendshape_scores[None])
                #exp *= 1.2
                #exp *= 0.4
                pose[0,3] += 0.03

                # Display the blendshape scores
                for i in range(52):
                    dpg.set_value(f"bs_{i}", blendshape_scores[i])

                # Get 68 dlib-format face landmarks
                face_landmarks = convert_landmarks_mediapipe_to_dlib(lmks_mp=lmks_dense)

                # Re-align image (tracking standard), this image will be used in our network model
                frame_rgb = image_align(frame_rgb, face_landmarks, output_size=min(WEBCAM_H, WEBCAM_W), 
                                        standard='tracking', padding_mode='constant')
                
                # Set camera pose
                rotation_vec, translation_vec = compute_head_pose_from_mp_landmarks_3d(face_landmarks=lmks_dense, img_h=img_h, img_w=img_w)
                yaw = min(0.3, max(-0.3, -500*rotation_vec[1][0] - yaw_realign))
                pitch = min(0.3, max(-0.3, 800*rotation_vec[0][0]-0.2 - pitch_realign))
                dpg.set_value("yaw", yaw)
                dpg.set_value("pitch", pitch)

                # Put the image to the buffer
                webcam_display_buffer[:, :] = frame_rgb / 255.

                # Update the Dear PyGui texture
                dpg.set_value("_texture_webcam", webcam_display_buffer)
            
                # Update avatar
                update_image()
        sleep(1)





"""
Callback Functions
"""
def open_folder_dialog():
    # Callback to open a native file dialog for folder selection with a default path
    global avatar_loaded
    global is_webcam_update
    is_webcam_update = False

    default_path = "./saved_avatars"
    root = Tk()
    root.withdraw()  # Hide the root tkinter window

    # Open folder selection dialog with the default path
    folder_selected = filedialog.askdirectory(initialdir=default_path)  
    
    # Update the path display in Dear PyGui
    if folder_selected:
        dpg.set_value("avatar_path", f"Avatar Path: \n{folder_selected}")
        
        # Check if the specific file exists in the selected folder
        file_to_check = "uv_delta_blendmaps.pt"
        if os.path.isfile(os.path.join(folder_selected, file_to_check)):
            print(f"The file '{file_to_check}' exists in the selected folder.")
            # load the head avatar
            dejavu.load_head_avatar(save_path=folder_selected, avatar_name='')
            avatar_loaded = True
            dpg.set_value("gaussians_text", len(dejavu.framework.uv_rasterizer.valid_coords[0]))
            update_image()
            is_webcam_update = True
        else:
            print(f"The file '{file_to_check}' does not exist in the selected folder.")
            dpg.set_value("avatar_path", f"File '{file_to_check}' not found in the given folder!")
            avatar_loaded = False

    root.destroy()  # Close the tkinter instance


def update_image():
    # Callback to update the texture with new image data
    global avatar_loaded, last_time, last_fps, to_blur

    if avatar_loaded == False:
        return

    # Set camera pose
    radius = dpg.get_value(f"radius")
    yaw = dpg.get_value(f"yaw")
    pitch = dpg.get_value(f"pitch")
    fov = dpg.get_value(f"fov"); dejavu.framework.fov = fov
    dx, dy, dz = compute_camera_offsets(yaw=yaw, pitch=pitch, radius=radius)
    camera_pose = np.array([[yaw,pitch,0,dx,dy,radius-dz]], dtype=np.float32)

    # Drive and render
    rendered = dejavu.drive_head_avatar(exp=exp[:,:50], pose=pose, eye_pose=eye_pose, cam_pose=camera_pose, return_all=False, exp_alpha=0.6)
    rendered = rendered[0].cpu().permute(1,2,0).numpy()

    # Blur the boundary
    if to_blur:
        rendered = blur_head_boundary(rendered_img=rendered)

    # Put the rendered image to display buffer
    display_buffer[DISPLAY_Y:DISPLAY_Y+RENDER_SIZE, DISPLAY_X:DISPLAY_X+RENDER_SIZE, :] = rendered

    # Update the texture data in Dear PyGui
    dpg.set_value("_texture", display_buffer)

    # Calculate FPS
    current_time = time()
    fps = 1.0 / (current_time - last_time)
    avg_fps = 0.6*fps + 0.4*last_fps
    last_fps = avg_fps
    last_time = current_time

    # Update the FPS display in Dear PyGui
    dpg.set_value("fps_text", f"FPS: {avg_fps:.2f}")


def realign_camera():
    global yaw_realign, pitch_realign
    yaw_realign = dpg.get_value("yaw")
    pitch_realign = dpg.get_value("pitch")
    dpg.set_value("radius", 1.2)
    dpg.set_value("fov", 20)
    update_image()  # Update the image after resetting sliders


def blur_checkbox_handler():
    global to_blur
    to_blur = not to_blur
    update_image()


def on_mouse_wheel(sender, app_data):
    global last_update_time
    delta = -1*app_data
    if dpg.is_item_hovered("_canvas_window"):
        radius = dpg.get_value(f"radius")
        SENSITIVITY = 0.01 # scroll wheel sensitivity
        radius = min(1.5, max(0.5, radius + delta * SENSITIVITY))
        dpg.set_value("radius", radius)
        current_time = time()
        if current_time - last_update_time > 0.05:
            update_image()
            last_update_time = current_time








"""
GUI
"""
# Create main window
dpg.create_context()
dpg.create_viewport(title='Gaussian Dejavu Head Avatar Real-time Driving', width=WINDOW_W+20, height=WINDOW_H+40)

# Create a texture for main display with initial empty data
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(WINDOW_W, WINDOW_H, display_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")


# Create main window with image display
with dpg.window(label="canvas", tag="_canvas_window", width=WINDOW_W, height=WINDOW_H+20, 
                no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True, no_resize=True):
    dpg.add_image("_texture", width=WINDOW_W, height=WINDOW_H, tag="_image")


# Create a texture for webcam display with initial empty data
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(WEBCAM_W, WEBCAM_H, webcam_display_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture_webcam")


# create webcam display window
with dpg.window(label="Webcam (Driving)", tag="_webcam_window", width=WEBCAM_W+15, height=WEBCAM_H+35, 
                pos=(WINDOW_W-275, WINDOW_H-460), no_resize=True):
    dpg.add_image("_texture_webcam", width=WEBCAM_W, height=WEBCAM_H, tag="_webcam" )



# Create a window for displaying Mediapipe blendshape scores
with dpg.window(label="Blendshape Scores", width=300, height=450, pos=(5, 5)):
    for i in range(52):
        dpg.add_slider_float(label=f"{i+1}", tag=f"bs_{i}",
                            default_value=0.0, min_value=0.0, max_value=1.0)


# Create a window for Camera sliders
with dpg.window(label="Camera", width=300, height=220, pos=(5, 465)):

    dpg.add_text("Rotate and Translate")
    dpg.add_slider_float(label=f"Left/Right", tag=f"yaw",
                                default_value=0.0, min_value=-0.3, max_value=0.3,
                                callback=lambda: update_image())
    dpg.add_slider_float(label=f"Up/Down", tag=f"pitch",
                                default_value=0.0, min_value=-0.3, max_value=0.3,
                                callback=lambda: update_image())
    dpg.add_slider_float(label=f"Distance", tag=f"radius",
                                default_value=1.2, min_value=0.5, max_value=1.5,
                                callback=lambda: update_image())
    
    dpg.add_text("Other")
    dpg.add_slider_float(label=f"FoV", tag=f"fov",
                                default_value=20, min_value=10, max_value=60,
                                callback=lambda: update_image())

    # Reset button
    dpg.add_text("")
    dpg.add_button(label="Realign", callback=lambda: realign_camera())



# Create a window for Render settings
with dpg.window(label="Render", width=200, height=100, pos=(WINDOW_W - 205, WINDOW_H - 105)):

    # Blur checkbox
    dpg.add_checkbox(label="Blur Boundary", tag="blur_checkbox", 
                     default_value=True, callback=lambda: blur_checkbox_handler())

    # Display current FPS
    dpg.add_text("FPS: 0.00", tag="fps_text")

    # Display number of gaussians
    dpg.add_text("Gaussians: 0", tag="gaussians_text")



# Create a window for avatar selector
with dpg.window(label="Avatar Selector", width=550, height=90, pos=(WINDOW_W - 555, 5)):
    # Button to open file dialog
    dpg.add_button(label="Load Avatar Model", callback=open_folder_dialog)
    # Avatar Path
    dpg.add_text("Please load the avatar model first!", tag="avatar_path")



# Global handlers
with dpg.handler_registry():
    dpg.add_mouse_wheel_handler(callback=on_mouse_wheel)







"""
Start Demo
"""

# Start the webcam capture in a separate thread
webcam_thread = threading.Thread(target=update_webcam_frame)
webcam_thread.start()

# Set up the initial loop callback
dpg.set_frame_callback(1, update_image)  # Start the update loop

# Launch the app
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()

# Exit
is_webcam_update = False # break the inner loop
is_webcam_running = False # break the outer loop
webcam_thread.join() # release the webcam thread
cap.release() # release webcam input stream
dpg.destroy_context() # release GUI resources

print('Program exited.')
