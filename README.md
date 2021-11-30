# ARKitScenes

This repo accompanies the research paper, [ARKitScenes - A Diverse Real-World Dataset for 3D Indoor Scene Understanding 
Using Mobile RGB-D Data](https://openreview.net/forum?id=tjZjv_qh_CE) and contains the data, scripts to visualize 
and process assets, and training code described in our paper.

![image](https://user-images.githubusercontent.com/7753049/144107932-39b010fc-6111-4b13-9c68-57dd903d78c5.png)

<img width="1418" alt="Screen Shot 2021-09-27 at 10 31 30" src="https://media.github.pie.apple.com/user/69097/files/25abd000-1f7e-11ec-8cda-d9b814e4418e">
<img width="1418" alt="Screen Shot 2021-09-27 at 10 35 38" src="https://media.github.pie.apple.com/user/69097/files/ad91da00-1f7e-11ec-9ef2-4d545c94a0d5">


## Paper
[ARKitScenes - A Diverse Real-World Dataset for 3D Indoor Scene Understanding 
Using Mobile RGB-D Data](https://openreview.net/forum?id=tjZjv_qh_CE)

upon using these data or source code, please cite
```buildoutcfg
@inproceedings{
dehghan2021arkitscenes,
title={{ARK}itScenes - A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile {RGB}-D Data},
author={Gilad Baruch and Zhuoyuan Chen and Afshin Dehghan and Tal Dimry and Yuri Feigin and Peter Fu and Thomas Gebauer and Brandon Joffe and Daniel Kurz and Arik Schwartz and Elad Shulman},
booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
year={2021},
url={https://openreview.net/forum?id=tjZjv_qh_CE}
}
```

## Overview
ARKitScenes is not only the first RGB-D dataset that is captured with now widely available depth sensor, but also is the 
largest indoor scene understanding data ever collected. In addition to the raw and processed data, ARKitScenes includes 
high resolution depth maps captured using a stationary laser scanner, as well as manually labeled 3D oriented bounding 
boxes for a large taxonomy of furniture. We further provide helper scripts for two downstream tasks: 
3D object detection and RGB-D guided upsampling. We hope that our dataset can help push the boundaries of 
existing state-of-the-art methods and introduce new challenges that better represent real world scenarios.

## Key features
• ARKitScenes is the first RGB-D dataset captured with the widely available
Apple LiDAR scanner. Along with the raw data we provide the camera pose and surface
reconstruction for each scene.

• ARKitScenes is the largest indoor 3D dataset consisting of 5,047 captures of 1,661 unique
scenes.

• We provide high quality ground truth of (a) registered RGB-D frames and (b) oriented
bounding boxes of room defining objects.

Below is an overview of RGB-D datasets and their ground truth assets compared with ARKitScenes.
HR and LR represent High Resolution and Low Resolution respectively, and are available for a subset of 2,257 captures of 841 unique
scenes.

<img width="815" alt="Screen Shot 2021-10-04 at 10 31 12" src="https://media.github.pie.apple.com/user/69097/files/3ddf9a00-24fe-11ec-8638-1e883c5a093f">


## Data collection

In the figure below, we provide  (a) illustration of iPad Pro scanning set up. (b) mesh overlay to assist data collection with iPad Pro. (c) example of one of the scan patterns captured with the iPad pro, the red markers show the chosen locations of the stationary laser scanner in that room.

<img width="999" alt="Screen Shot 2021-09-29 at 23 12 07" src="https://media.github.pie.apple.com/user/69097/files/b2bb7880-217a-11ec-9e7c-b23dce063c3f">

## Data download

To download the data please follow the [data](DATA.md) documentation
 
## Tasks

Here we provide the two tasks mentioned in our paper, namely, 3D Object Detection (3DOD) and depth upsampling.

### [3DOD](threedod/README.md)

### [Depth upsampling](depth_upsampling/README.md)

## License
This dataset is released with non-commercial creative commons license.
