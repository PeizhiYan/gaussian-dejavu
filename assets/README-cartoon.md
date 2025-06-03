# Gaussian Deja-vu 🐈‍⬛ ✖️ VToonify: Cartoon Head Avatar

[Return](../README.md)

## 📺 Demo Video (Click to Watch)
<div align="center">
  <table>
    <tr>
      <td>
        <a href="https://www.youtube.com/watch?v=Drq4I-8QRcc">
          <img src="https://img.youtube.com/vi/Drq4I-8QRcc/0.jpg" alt="Video">
        </a>
      </td>
    </tr>
  </table>
</div>





### 1️⃣ Prepare Video

When collecting your video, please consider following this guidance to achieve good reconstruction results [```./assets/personal_video_collection_procedure.pdf```](../assets/personal_video_collection_procedure.pdf)




### 2️⃣ Stylize Video (VToonify)

- First, follow VToonify repo to create the virtual environment: https://github.com/williamyang1991/VToonify
- Then, run ```style_transfer.py``` to stylize the video. Example command:
```bash
python style_transfer.py --content /mnt/data3_hdd/peizhi/Datasets/Personal_Videos/peizhi-demo/IMG_3198.mov --video \
       --scale_image --style_id 2 --style_degree 0.5 \
       --ckpt ./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt \
       --padding 600 600 600 600     # use large padding to avoid cropping the image
```
- The stylized video will be saved to ```output``` folder.



### 3️⃣ Prepare Training Data

Please follow https://github.com/PeizhiYan/flame-head-tracker/tree/v3.2 and our example [```./examples/Personal-Video-Precessing.ipynb```](../examples/Personal-Video-Processing.ipynb) to pre-process your stylized video.

**Note that, ```Dejavu v1.1``` code is compatible with ```flame-head-tracker v3.2```, not higher versions.**



### 4️⃣ Train Cartoon Head Avatar

Please follow this notebook to train the avatar: [```./examples/Personalize-Avatar-cartoon.ipynb```](../examples/Personalize-Avatar-cartoon.ipynb)

- We have a pre-trained avatar model you can load in the Viewer to try: https://github.com/PeizhiYan/gaussian-dejavu/releases/download/v1.1/peizhi-cartoon-uv320-v1.1.zip









---

### Avatar Viewer Demo

```
python run_avatar_viewer.py
```

We have prepared some head avatar models in the folder ```./saved_avatars/```. Please note that, ```imavatar``` models were trained on the IMAvatar dataset (https://github.com/zhengyuf/IMavatar). 


### Realtime Avatar Webcam-Driving Demo

```
python run_avatar_driver.py
```






