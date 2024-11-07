# Gaussian Deja-vu 🐈‍⬛

## [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- **Author**: Peizhi Yan
- **Date Updated**: 11-06-2024


## 📺 Demo Video (Click to Watch)
[![Watch the video](https://img.youtube.com/vi/Tm7uPEYzfpo/0.jpg)](https://www.youtube.com/watch?v=Tm7uPEYzfpo)


**Notations**:
⭐ Important 
✨ Idea
❓ Question





## 💚 Citation

This is the official code repo for our "Gaussian Deja-vu" (accepted for WACV 2025 in Round 1). 

Please consider citing our work if you find this code useful.
```
@article{yan2024gaussian,
  title={Gaussian Deja-vu: Creating Controllable 3D Gaussian Head-Avatars with Enhanced Generalization and Personalization Abilities},
  author={Yan, Peizhi and Ward, Rabab and Tang, Qiang and Du, Shan},
  journal={arXiv preprint arXiv:2409.16147},
  year={2024}
}
```



## 🌱 Todos
- [x] Avatar viewer demo.
- [ ] Test on another computer with Ubuntu.
- [ ] Video head avatar driving demo.
- [ ] Test on Windows system.





## 🟠 Environment

[Return](#)

### Prerequisites:

- **GPU**: Nvidia GPU with >= 6GB memory (recommend > 8GB). I tested the code on Nvidia A6000 (48GB) GPU.
- **OS**: Ubuntu Linux (tested on 22.04 LTS and 24.04 LTS), I haven't tested the code on Windows.

### Step 1: Create a conda environment. 

```
conda create --name dejavu -y python=3.10
conda activate dejavu
```

### Step 2: Install necessary libraries.

#### Nvidia CUDA compiler (11.7)

```
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja

# (Linux only) ----------
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"

# Install NVCC (optional, if the NVCC is not installed successfully try this)
conda install -c conda-forge cudatoolkit=11.7 cudatoolkit-dev=11.7
```

After install, check NVCC version (should be 11.7):

```
nvcc --version
```

#### PyTorch (2.0 with CUDA)

```
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu117
```

Now let's test if PyTorch is able to access CUDA device, the result should be ```True```:

```
python -c "import torch; print(torch.cuda.is_available())"
```

#### Some Python packages

```
pip install -r requirements.txt
```

#### Nvidia Differentiable Rasterization: nvdiffrast

**Note that**, we use nvdiffrast version **0.3.1**, other versions may also work but not promised.

```
# Download the nvdiffrast from their official Github repo
git clone https://github.com/NVlabs/nvdiffrast

# Go to the downloaded directory
cd nvdiffrast

# Install the package
pip install .

# Change the directory back
cd ..
```

#### Pytorch3D

**Note that**, we use pytorch3d version **0.7.8**, other versions may also work but not promised.

Installing pytorch3d may take a bit of time.

```
# Download Pytorch3D from their official Github repo
git clone https://github.com/facebookresearch/pytorch3d

# Go to the downloaded directory
cd pytorch3d

# Install the package
pip install .

# Change the directory back
cd ..
```

#### Troubleshoot

Note that the NVCC needs g++ < 12:
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 50
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 50
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-11 50
```

If there is problem with **nvdiffrast**, check whether it is related to the EGL header file in the error message. If it is, install the EGL Development Libraries (for Ubuntu/Debian-based systems):
```
sudo apt-get update
sudo apt-get install libegl1-mesa-dev
```
Then, uninstall nvdiffrast and reinstall it.


### Step 3: Download some necessary model files.

⭐ Because of **copyright concerns**, we cannot re-share any of the following model files. Please follow the instructions to download the necessary model file.

- Download ```FLAME 2020 (fixed mouth, improved expressions, more data)``` from https://flame.is.tue.mpg.de/ and extract to ```./models/FLAME2020```

- Download ```face_landmarker.task``` from https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task, rename as ```face_landmarker_v2_with_blendshapes.task```, and save at ```./models/```














