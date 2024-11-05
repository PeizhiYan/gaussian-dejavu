# Project DejaVu Development Notes

+ **Author**: Peizhi Yan
+ **Created on**: Feb. 20, 2024
+ **Last Modified on**: May. 23, 2024
+ **Current Version**: 3.2

⭐ Important/Milestone 
✨ Idea
❓ Question
✅ Done

## Table of Contents
- 🔴 [FLAME Model Description](#🔴-flame-model-description)
- 🟠 [Environment](#🟠-environment)
- 🟡 [Folder Structure](#🟡-folder-structure)
- 🟢 [Major Todos](#🟢-major-todos)
- 🔵 [Dev. Logs](#🔵-dev-logs)
- 🟣 [xxx]()

### Tags

- 🟧 Note
- 🟨 Bug / Issue
- 🟩 Solution
- 🟦 Investigation








## 🔴 FLAME Model Description

[Return](#table-of-contents)

- We ues **FLAME 2020**, same as in DECA. **Note**: using other FLAME versions will cause problems. For example, the issue identified with using FLAME 2023 is that the expressions are not correct.
- **FLAME Tracking Results (We Use)**:
    - **vertices**: [5023,3] 
    - **shape**: [1,100] 
    - **exp**: [1,50]
    - **tex**: [1,50]       (to generate albedo map using FLAME texture model)
    - **pose**: [6]       (first three for __*head pose*__, last three for __*jaw pose*__)
    - **light**: [1,9,3]    (__*SH coefficients*__)
    - **cam**: [6]        (6DoF yaw,pitch,roll,x,y,z offsets)
    - **uv_texture**: [256,256,3]      (UV texture map, generated from original image using DECA, **uint8** format, 0~255, **Note**: this is different from the albedo map)
    - **parsing**: [512,512]      
    - **img_aligned**: [1024,1024]      
    - **img_rendered**: [256,256,3]      









## 🟠 Environment

[Return](#table-of-contents)


The important part is to compile the differentiable Gaussian splatting renderer. The easiest way is to follow GaussianAvatars, to do it in a conda virtual environment:

```
git clone https://github.com/ShenhanQian/GaussianAvatars.git --recursive
cd GaussianAvatars

conda create --name dejavu -y python=3.10
conda activate dejavu

# Install CUDA and ninja for compilation
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja  # use the right CUDA version
# (Linux only) ----------
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"
# (Windows only) --------
conda env config vars set CUDA_PATH="$env:CONDA_PREFIX"  # re-activate the environment to make effective
# ---------------------

# Install PyTorch (make sure the CUDA version match with the above)
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu117
# or
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
# make sure torch.cuda.is_available() returns True

# Install the rest pacakges (can take a while for compiling pytorch3d and nvdiffrast)
pip install -r requirements.txt
```

After the creation of conda environment, go to the ```diff-gaussian-rasterization``` folder, run the setup tool:
```
python setup.py install

# alternatively
pip install .
```

### Tested on the following setting (Ubuntu Linux 22.04.2 LTS):
- Python 3.10.13
- NVCC (CUDA) 11.7
- Pytorch 2.0.1+cu117











## 🟡 Folder Structure

[Return](#table-of-contents)


### *root*

- ```DejaVu-Dev/```: 

### dev-notebooks

Development playground notebooks.

- ```DejaVu-Dev/dev-notebooks/ffhq_to_flame.ipynb```: To test the preprocessing pipeline for estimating the FLAME parameters for the FFHQ images.

### external

External project repos. **We isolate this folder to our project code.** If some components needed in our project, we copy them to our folder.

- ```DejaVu-Dev/external/BFM_to_FLAME```: to generate the ```FLAME_albedo_from_BFM.npz``` file.
- ```DejaVu-Dev/external/DECA```: DECA face reconstruction.
- ```DejaVu-Dev/external/DECA/vertices_to_uv_coords.ipynb```: to generate the correspondences between 2D UV texture coordinates and 3D mesh vertex indices.
- ```DejaVu-Dev/external/FLAME```: FLAME original repo. Contains FLAME models 2019, 2020, 2023.
- ```DejaVu-Dev/external/FLAME_photometric_fitting```: FLAME photometric fitting codes. 

### models

- ```DejaVu-Dev/models/FLAME2020```
- ```DejaVu-Dev/models/FLAME_texture.npz```
- ```DejaVu-Dev/models/head_template.obj```
- ```DejaVu-Dev/models/landmark_embedding.npy```
- ```DejaVu-Dev/models/uv2vert_256.npy```

### utils

Utility functions and tools.

- ```DejaVu-Dev/utils/o3d_```: Open3D utility functions. 

### scripts

- ⭐ ```DejaVu-Dev/scripts/ffhq_to_flame.py```: This is a dataset **pre-processing** tool, to estimate the FLAME parameters of the FFHQ images.







