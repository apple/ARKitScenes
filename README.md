# ARKitScenes

This repo accompanies the research paper, [ARKitScenes - A Diverse Real-World Dataset for 3D Indoor Scene Understanding 
Using Mobile RGB-D Data](https://openreview.net/forum?id=tjZjv_qh_CE) and contains the data, scripts to visualize 
and process assets, and training code described in our paper.

![image](https://user-images.githubusercontent.com/7753049/144107932-39b010fc-6111-4b13-9c68-57dd903d78c5.png)

![image](https://user-images.githubusercontent.com/7753049/144108052-6a1d3a67-3948-4ded-bd08-6f1572fdf97a.png)

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

![image](https://user-images.githubusercontent.com/7753049/144108117-b789a5be-cc08-44f0-a76c-f1549c59825e.png)


## Data collection

In the figure below, we provide  (a) illustration of iPad Pro scanning set up. (b) mesh overlay to assist data collection with iPad Pro. (c) example of one of the scan patterns captured with the iPad pro, the red markers show the chosen locations of the stationary laser scanner in that room.

![image](https://user-images.githubusercontent.com/7753049/144108161-0ae7ba6a-305f-4a22-93b1-0b2d1e78154e.png)

## Data download

To download the data please follow the [data](DATA.md) documentation
 
## Tasks

Here we provide the two tasks mentioned in our paper, namely, 3D Object Detection (3DOD) and depth upsampling.

### [3DOD](threedod/README.md)

### [Depth upsampling](depth_upsampling/README.md)

## License
The ARKitScenes dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/.
For queries regarding a commercial license, contact ARKitScenes-license@group.apple.com
If you have any other questions raise an issue in the repository and contact ARKitScenes@group.apple.com
