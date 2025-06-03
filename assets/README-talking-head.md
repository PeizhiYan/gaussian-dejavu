# Gaussian Deja-vu 🐈‍⬛ ✖️ UniTalker: Talking Head Demo

[Return](../README.md)

## 📺 Demo Video (Click to Watch)
<div align="center">
  <table>
    <tr>
      <td>
        <a href="https://www.youtube.com/watch?v=vGlJ-Qzg818">
          <img src="https://img.youtube.com/vi/vGlJ-Qzg818/0.jpg" alt="Video">
        </a>
      </td>
    </tr>
  </table>
</div>





### 1️⃣ Train the Head Avatar

Please follow [README](../README.md) to train the head avatar model.




### 2️⃣ Convert Audio to FLAME Parameters (using UniTalker)

- Clone the UniTalker repo: https://github.com/X-niper/UniTalker/tree/main

- Create a virtual environment for UniTalker:
  ```bash
    conda create -n unitalker python==3.10
    conda activate unitalker
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install transformers=4.39.3 librosa tensorboardX smplx chumpy numpy==1.23.5 opencv-python
  ```

- Copy ```./examples/unitalker-to-flame.ipynb``` from our project repo to the folder of UniTalker code.

- Run the code in ```unitalker-to-flame.ipynb``` notebook to convert your audio to the ```.npy``` file.
  - We have an example converted file stored at ```./assets/can_you_feel_the_love_tonight_clip.npy```.



### 3️⃣ Render and Export Video

- Follow ```./examples/Talking-Head.ipynb``` to render and export the video.







