#
# Gaussian DejaVu Head Avatar Viewer
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

from dejavu import GaussianDejavu
from utils.personal_video_utils import *


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

    # Create alpha mask for blending
    alpha = cv2.GaussianBlur(eroded_mask.astype(float) / 255.0, (blur_kernel_size, blur_kernel_size), sigmaX=sigma*2)

    # Blend the original image with blurred boundary
    blurred_img = (alpha[..., None] * blurred_boundary + (1 - alpha[..., None]) * rendered_img)

    return blurred_img / 255.







"""
Create Gaussian Dejavu Pipeline
"""
dejavu = GaussianDejavu(network_weights='./models/dejavu_network.pt', uv_map_size=120, num_expressions=20)
device = dejavu.device







"""
GUI Parameters
"""
WINDOW_H = 700
WINDOW_W = 1024
RENDER_SIZE = dejavu.framework.H
DISPLAY_Y = (WINDOW_H - RENDER_SIZE) // 2
DISPLAY_X = (WINDOW_W - RENDER_SIZE) // 2


"""
Global Variables
"""
avatar_loaded = True ; dejavu.load_head_avatar(save_path='./saved_avatars', avatar_name='peizhi-uv120')
#avatar_loaded = False
display_buffer = np.ones((WINDOW_H, WINDOW_W, 3), dtype=np.float32)
last_time = time()  # Initialize the last time for FPS calculation
last_fps = 0
to_blur = True
prev_x, prev_y = None, None
is_dragging_in_canvas = False
last_update_time = time()




"""
Callback Functions
"""
def open_folder_dialog():
    # Callback to open a native file dialog for folder selection with a default path
    global avatar_loaded

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

    # Adjust FLAME expression coefficients
    exp = np.zeros([1,50], dtype=np.float32)
    for i in range(10): exp[0,i] = dpg.get_value(f"expression_{i}")

    # Adjust FLAME jaw pose
    pose = np.zeros([1,6], dtype=np.float32)
    pose[0,3] = dpg.get_value(f"jaw_UD")
    pose[0,4] = dpg.get_value(f"jaw_LR")

    # Adjust FLAME eye pose
    eye_pose = np.zeros([1,6], dtype=np.float32)
    eye_pose[0, 1] = dpg.get_value(f"eyes") # left eye
    eye_pose[0, 4] = dpg.get_value(f"eyes") # right eye

    # Set camera pose
    radius = dpg.get_value(f"radius")
    yaw = dpg.get_value(f"yaw")
    pitch = dpg.get_value(f"pitch")
    fov = dpg.get_value(f"fov"); dejavu.framework.fov = fov
    dx, dy, dz = compute_camera_offsets(yaw=yaw, pitch=pitch, radius=radius)
    camera_pose = np.array([[yaw,pitch,0,dx,dy,radius-dz]], dtype=np.float32)

    # Drive and render
    rendered = dejavu.drive_head_avatar(exp=exp, pose=pose, eye_pose=eye_pose, cam_pose=camera_pose, return_all=False)
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


def reset_flame_sliders():
    # Reset expression sliders
    for i in range(10):
        dpg.set_value(f"expression_{i}", 0.0)
    # Reset jaw and eyes sliders
    dpg.set_value("jaw_UD", 0.04)
    dpg.set_value("jaw_LR", 0.00)
    dpg.set_value("eyes", 0.0)
    update_image()  # Update the image after resetting sliders


def reset_camera_sliders():
    dpg.set_value("yaw", 0.0)
    dpg.set_value("pitch", 0.00)
    dpg.set_value("radius", 1.2)
    dpg.set_value("fov", 20)
    update_image()  # Update the image after resetting sliders


def blur_checkbox_handler():
    global to_blur
    to_blur = not to_blur
    update_image()


def on_drag_start(sender, app_data):
    global last_mouse_pos, is_dragging_in_canvas
    if dpg.is_item_hovered("_canvas_window"):  # Check if mouse is over canvas window
        is_dragging_in_canvas = True
        last_mouse_pos = dpg.get_mouse_pos()
        # print('STARTED')


def on_drag(sender, app_data):
    global last_mouse_pos, is_dragging_in_canvas, last_update_time
    #current_x, current_y = app_data[1], app_data[2]
    
    if not is_dragging_in_canvas:
        return

    current_mouse_pos = dpg.get_mouse_pos()
    if last_mouse_pos is None:
        last_mouse_pos = current_mouse_pos
    else:
        # compute the mouse move offsets
        delta_x = current_mouse_pos[0] - last_mouse_pos[0]
        delta_y = current_mouse_pos[1] - last_mouse_pos[1]

        yaw = dpg.get_value("yaw")
        pitch = dpg.get_value("pitch")

        SENSITIVITY = 0.0016
        yaw = min(0.3, max(-0.3, yaw + delta_x * SENSITIVITY))
        pitch = min(0.3, max(-0.3, pitch + delta_y * SENSITIVITY))

        dpg.set_value("yaw", yaw)
        dpg.set_value("pitch", pitch)

        # Check the time since the last update to limit the frequency of updates
        current_time = time()
        if current_time - last_update_time > 0.05:
            update_image()
            last_update_time = current_time

        last_mouse_pos = current_mouse_pos # update previous locations


def on_drag_stop(sender, app_data):
    global last_mouse_pos, is_dragging_in_canvas
    is_dragging_in_canvas = False
    last_mouse_pos = None
    # print('STOPED')


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
dpg.create_viewport(title='Gaussian Dejavu Head Avatar Viewer', width=WINDOW_W+20, height=WINDOW_H+40)

# Create a texture with initial empty data
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(WINDOW_W, WINDOW_H, display_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")


# Create main window with image display
with dpg.window(label="canvas", tag="_canvas_window", width=WINDOW_W, height=WINDOW_H+20, 
                no_title_bar=True, no_move=True, no_bring_to_front_on_focus=True, no_resize=True):
    dpg.add_image("_texture", width=WINDOW_W, height=WINDOW_H, tag="_image")


# Create a window for FLAME sliders
with dpg.window(label="FLAME", width=300, height=450, pos=(5, 5)):

    # Expression coefficients
    dpg.add_text("Expression Coefficients")
    for i in range(10):
        dpg.add_slider_float(label=f"{i}", tag=f"expression_{i}",
                             default_value=0.0, min_value=-1.5, max_value=1.5,
                             callback=lambda: update_image())

    # Jaw up/down
    dpg.add_text("Jaw")
    dpg.add_slider_float(label=f"Up/Down", tag=f"jaw_UD",
                                default_value=0.04, min_value=0, max_value=0.3,
                                callback=lambda: update_image())
    
    # Jaw left/right
    dpg.add_slider_float(label=f"Left/Right", tag=f"jaw_LR",
                                default_value=0.0, min_value=-0.1, max_value=0.1,
                                callback=lambda: update_image())

    # Eyes
    dpg.add_text("Eyes")
    dpg.add_slider_float(label=f"Left/Right", tag=f"eyes",
                                default_value=0.0, min_value=-0.3, max_value=0.3,
                                callback=lambda: update_image())

    # Reset button
    dpg.add_text("")
    dpg.add_button(label="Reset", callback=lambda: reset_flame_sliders())



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
    dpg.add_button(label="Reset", callback=lambda: reset_camera_sliders())



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
    dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=on_drag_start)
    dpg.add_mouse_drag_handler(callback=on_drag, threshold=1)
    dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=on_drag_stop)
    dpg.add_mouse_wheel_handler(callback=on_mouse_wheel)







"""
Start Demo
"""

# Set up the initial loop callback
dpg.set_frame_callback(1, update_image)  # Start the update loop

# Launch the app
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

