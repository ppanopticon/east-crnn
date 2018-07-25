# EAST + CRNN

### Introduction
This is an end-to-end scene text detection demo based on [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2) and [CRNN](http://arxiv.org/abs/1507.05717). It is basically a combination
of the work done by [argman](https://github.com/argman/EAST) and [MaybeShewill-CV](https://github.com/MaybeShewill-CV/CRNN_Tensorflow).

### Contents
1. [Installation](#installation)
2. [Download](#download)
2. [Run demo server](#demo)

### Installation
1. Any version of Tensorflow version > 1.0 should be ok.

### Download
1. Pre-trained model by EAST author, see [BaiduYun link](http://pan.baidu.com/s/1jHWDrYQ) or [GoogleDrive](https://drive.google.com/open?id=0B3APw5BZJ67ETHNPaU9xUkVoV0U)
2. Pre-trained CRNN model by original author see [GitHub](https://github.com/MaybeShewill-CV/CRNN_Tensorflow/tree/master/model/shadownet)

### Run Demo Server
If you've downloaded the pre-trained model, you can setup a demo server 
```
python3 run_demo_server.py 
```
Then open http://localhost:8769 for the web demo. Notice that the URL will change after you submitted an image.
Something like `?r=49647854-7ac2-11e7-8bb7-80000210fe80` appends and that makes the URL persistent.
As long as you are not deleting data in `static/results`, you can share your results to your friends using
the same URL.

URL for example below: http://east.zxytim.com/?r=48e5020a-7b7f-11e7-b776-f23c91e0703e
![web-demo](demo_images/web-demo.png)