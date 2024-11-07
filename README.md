# Gaussian Deja-vu

[![Watch the video](https://img.youtube.com/vi/Tm7uPEYzfpo/0.jpg)](https://www.youtube.com/watch?v=Tm7uPEYzfpo)


**Notations**:
⭐ Important/Milestone 
✨ Idea
❓ Question












## 🟠 Environment

[Return](#)


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















