# 3DOD tutorials

# Data download
To download the data please follow the [data](../DATA.md) documentation

# Code structure

**The code is structured as follows**
```python
threedod/
    benchmark_scripts/
        data_prepare_offline.py     # scripts to prepare whole-scan dataset
        data_prepare_online.py      # scripts to prepare single-frame dataset
        demo_eval.py                # demo scripts to evaluate mAP and recall
        prepare_votes.py            # scripts to prepare votes supervision for some sota approaches (votenet, h3dnet, mlcvnet)
        download_3dod_data.py       # scripts to download 3dod dataset
        show_3d_bbox_annotation.py  # scripts to show visualization of box with point cloud
        
        utils/    # folders with utility functions
            box_utils.py            # functions related to manipulating 3D boxes
            pc_utils.py             # functions related to point clouds
            rotation.py             # all rotation related stuff
            eval_utils.py           # functions related to mAP, recall evaluation
            visual_utils.py         # visualization
            tenFpsDataLoader.py     # dataloader for the raw data
            taxonomy.py             # categories we are training
```

# Creating a virtual environment

Run below command to install relevant packages. This will create a venv/ folder in base directory
```
bash python_venv_setup.sh
```

Then activate the virtual environment we have created:
```
source venv/bin/activate
```


# Data organization and format of input data

```
ARKitScenes/threedod/sample_data/40753679/
├── 40753679_3dod_annotation.json
├── 40753679_3dod_mesh.ply
└── 40753679_frames
    ├── color.traj               # camera poses
    ├── color_intrinsics         # camera intrinsics
    │   ├── 6845.7061.pincam     # filenames are indexed by timestamps
    │   ├── 6845.8060.pincam
    │   ├── 6845.9060.pincam
    │   └── ...
    ├── depth_densified          # depth frames
    │   ├── 6845.80601079.png    # filenames are indexed by timestamps
    │   ├── 6845.90596450.png
    │   └── ...
    └── wide                     # color frames
        ├── 6845.70605696.png    # filenames are indexed by timestamps
        ├── 6845.80601079.png
        ├── 6845.90596450.png
        └── ...
```

# Visualizing 3DOD annotations

To view 3d bounding box annotations on a mesh, you can follow below steps:

Unzip downloaded data into `ARKitScenes/threedod/sample_data` directory

```buildoutcfg
ARKitScenes/threedod/sample_data
├── 47331606
├── 41069022
└── 41069023
```

Then run below script:
```
python3 threedod/benchmark_scripts/show_3d_bbox_annotation.py \
        -f threedod/sample_data/47331606/47331606_3dod_mesh.ply \
        -a threedod/sample_data/47331606/47331606_3dod_annotation.json
```

A vtk screen like below will pop up, showing our annotated 3d bounding boxes drawn on a color mesh.

<img width="600" alt="Screen Shot 2021-09-29 at 22 17 43" src="https://media.github.pie.apple.com/user/69097/files/3ffacf00-2173-11ec-9966-2ca5f8e7757d">


# Annotation file format
For each scan, an annotation json file is provided.

```buildoutcfg
|-- data[]: list of bounding box data
|  |-- label: object name of bounding box
|  |-- axesLengths[x, y, z]: size of the origin bounding-box before transforming
|  |-- centroid[]: the translation matrix（1*3）of bounding-box
|  |-- normalizedAxes[]: the rotation matrix（3*3）of bounding-box 
```

# Preparing whole scene (offline) data with visualizations

Run below command:
```sh
cd ./threedod/benchmark_scripts
python3 data_prepare_offline.py \
        --data_root ../sample_data/ \
        --scene_id 47331606 \
        --gt_path ../sample_data/47331606/47331606_3dod_annotation.json \
        --output_dir ../sample_data/offline_prepared_data/ \
        --vis
```
The codes go through the video and accumulate the point cloud in each frame with a consistent coordinate system ("world coordinate") by leveraging intrinsic and extrinsic camera information. The label is in the same "world coordinate".

Below is a sample visualization.

<img width="579" alt="Screen Shot 2021-11-02 at 12 03 10" src="https://media.github.pie.apple.com/user/69097/files/2e27a400-3bd7-11ec-9642-e1c6feb487b1">


The command will output two folders for point cloud data and label. Notice the size of the bounding boxes saved is full-size rather than half-size as in votenet codes.
```buildoutcfg
../sample_data/offline_prepared_data
├── 47331606_data
    └─47331606_pc.npy
└── 47331606_label
    └─47331606_bbox.npy
```
The data and label can be opened with 
```python
import numpy as np
data = np.load("47331606_pc.npy")
label = np.load("47331606_bbox.npy", allow_pickle=True).item()
```
the former is a N by 3 numpy array; the latter is a dictionary with 4 keys for all m oriented bounding boxes: 
```
gt_labels = {
        "bboxes": # (m, 7) array, 7 dimension as (x, y, z, dx, dy, dz, theta)
        "types":  # list of length m, string for category name
        "uids":  # list of length m, sting for box id (optional)
        "pose":  # list of poses in each frame (optional)
}
```
the 7-digit codes for the oriented bounding boxes can be transferred to 8 corners by function `boxes_to_corners_3d()` in ./threedod/threedod_scripts/utils/box_utils.py

# Preparing single frame (online) data with visualizations

Run below command:
```sh
cd ./threedod/benchmark_scripts
python3 data_prepare_online.py \
        --data_root ../sample_data/ \
        --gt_path ../sample_data/47331606/47331606_3dod_annotation.json \
        --scene_id 47331606 \
        --output_dir ../sample_data/online_prepared_data/ \
        --vis
```

Below is a sample visualization.

<img width="614" alt="Screen Shot 2021-11-02 at 12 04 24" src="https://media.github.pie.apple.com/user/69097/files/47c8eb80-3bd7-11ec-9136-f88414c6aa72">


The command will output two folders for point cloud data and label.

```buildoutcfg
../sample_data/offline_prepared_data
├── 40753679_data
    └─40753679_0_pc.npy
    └─40753679_2_pc.npy
    └─40753679_4_pc.npy
    └─...
└── 40753679_label
    └─40753679_0_bbox.npy
    └─40753679_2_bbox.npy
    └─40753679_4_bbox.npy
    └─...
```

## Preprocessing: To Prepare Votes for VoteNet/H3DNet/MLCVNet Training
After we get the point cloud data and box label, we provide the following example code we use to prepare for the votes supervision:
```sh
cd ./threedod/benchmark_scripts
python3 prepare_votes.py --vis
```

As in our paper, we evaluated three state-of-the-art approaches: [VoteNet](https://github.com/facebookresearch/votenet), [H3DNet](https://github.com/zaiweizhang/H3DNet), [MLCVNet](https://github.com/NUAAXQ/MLCVNet). And we reported their performance of whole-scan 3D objection detection.

| Whole-Scan | VoteNet  | H3DNet  | MLCVNet  |
| :------:   | :------: | :-----: | :------: |
| mAP        | 0.358    | 0.383   | 0.419    |

## Disclaimer
This codebase referred to several open-source projects. Some box generation and rotation functions are from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
For evaluating the mean average precision and average recall, we referred to [votenet](https://github.com/facebookresearch/votenet) as well as the [VOC-evalution](https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py).
