#TODO: remove sys.path based import

#!/usr/bin/env python3

import argparse
import numpy as np
import os
import time

import utils.box_utils as box_utils
import utils.rotation as rotation
from utils.taxonomy import class_names, ARKitDatasetConfig
from utils.tenFpsDataLoader import TenFpsDataLoader, extract_gt
import utils.visual_utils as visual_utils

dc = ARKitDatasetConfig()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="../sample_data/",
        help="input folder with ./scene_id"
        "extracted by unzipping scene_id.tar.gz"
    )
    parser.add_argument(
        "--scene_id",
        default="47331606",
    )
    parser.add_argument(
        "--gt_path",
        default="../../threedod/sample_data/47331606/47331606_3dod_annotation.json",
        help="gt path to annotation json file",
    )
    parser.add_argument("--frame_rate", default=1, help="sampling rate of frames")
    parser.add_argument(
        "--output_dir", default="../sample_data/online_prepared_data/", help="directory to save the data and annoation"
    )
    parser.add_argument("--vis", action="store_true")

    args = parser.parse_args()

    # step 0.1: get annotation first,
    # if skipped or no gt, we will not bother calling further steps
    gt_fn = args.gt_path
    skipped, boxes_corners, centers, sizes, labels, uids = extract_gt(gt_fn)
    if skipped or boxes_corners.shape[0] == 0:
        exit()
    n_gt = boxes_corners.shape[0]
    label_type = np.array([labels, uids])

    # step 0.2: data
    data_path = os.path.join(args.data_root, args.scene_id, f"{args.scene_id}_frames")
    print(os.path.abspath(data_path))
    loader = TenFpsDataLoader(
        dataset_cfg=None,
        class_names=class_names,
        root_path=data_path,
    )

    # step 0.3: output folder, make dir
    output_data_dir = os.path.join(args.output_dir, "%s_data" % args.scene_id)
    output_label_dir = os.path.join(args.output_dir, "%s_label" % args.scene_id)
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    t = time.time()

    world_pc, world_rgb = [], []
    args.frame_rate = max(2, len(loader) // 300)
    total_mask = []
    for i in range(len(loader)):
        frame = loader[i]
        print(i, frame["image_path"])
        image_path = frame["image_path"]
        frame_id = image_path.split(".")[-2]
        frame_id = frame_id.split("_")[-1]

        # step 2.1 get data accumulated to current frame
        # in upright camera coordinate system
        image = frame["image"]
        pcd = frame["pcd"] # in world coordinate
        pose = frame["pose"]
        rgb = frame["color"]
        urc, urc_inv = rotation.upright_camera_relative_transform(pose)

        # in case we jump frame with args.frame_rate > 1
        if i % args.frame_rate != 0:
            continue

        # rotate pcd to urc coordinate
        pcd = rotation.rotate_pc(pcd, urc)

        # 2. gt_boxes
        # 2.1 get all boxes to urc coord
        boxes_corners_urc = rotation.rotate_pc(
            boxes_corners.reshape(-1, 3), urc
        ).reshape(-1, 8, 3)
        # 2.2 get box codes
        boxes = box_utils.corners_to_boxes(boxes_corners_urc)
        # 2.3 apply a simple box-filter by removing boxes with < 10 points
        # (n, m)
        mask_pts_in_box = box_utils.points_in_boxes(pcd, boxes_corners_urc)
        # (m, )
        pts_cnt = np.sum(mask_pts_in_box, axis=0)
        mask_box = pts_cnt > 20
        gt_labels = {
            "bboxes": boxes,
            "bboxes_mask": mask_box,
            "types": labels,
            "uids": uids,
            "pose": pose,
        }

        # save points and boxes
        # ./output_dir/{scene_id}_data/xxx_x_pc.npy
        data_save_fn = "%s_%d_pc.npy" % (args.scene_id, i)
        data_save_fn = os.path.join(output_data_dir, data_save_fn)
        np.save(data_save_fn, pcd)
        # ./output_dir/{scene_id}_label/xxx_x_bbox.npy
        box_save_fn = "%s_%d_bbox.npy" % (args.scene_id, i)
        box_save_fn = os.path.join(output_label_dir, box_save_fn)
        np.save(box_save_fn, gt_labels)

        # 4. optional: visualize to see if it is correct
        if args.vis:
            # 1. points and boxes align
            corners_recon = box_utils.boxes_to_corners_3d(boxes)
            corners_recon = corners_recon[mask_box]
            visual_utils.visualize_o3d(pcd, corners_recon)

    elapased = time.time() - t
    print("total time: %f sec" % elapased)
