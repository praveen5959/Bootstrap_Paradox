**Problem :**
To filter out the background during the live stream of educational videos

**Solution:**
Given the parameters and constraints presented by the problem, in terms of quality of the output, speed (in terms of fps) of the output, and cost of the resources required (CPU/GPU cost) , we chose to implement a version DeepLabV3 which is based on the encoder decoder architecture. The video if fed to the model frame by frame and the primary person (educator) is identified and segmented. The background is subtracted in this way from every frame and the output frames are compressed together to form the video which would not have any background objects or people. Implementing DeepLabV3 also ensures that the latency caused by this processing is reduced to a fraction of a second. 

**Model Architecture**
![alt text](https://www.oreilly.com/library/view/hands-on-image-processing/9781789343731/assets/d929fc98-db73-4c77-9d60-4b4582fa03e2.png)

**Instructions to run:**
1. Run `sudo pip3 install -r requirements.txt` to install all packages.
2. Run  `python inference_video.py` for video files.
3. Run `python inference_webcam.py` for live streaming.


