#TODO: remove sys.path based import

#!/usr/bin/env python3

import argparse
import numpy as np
import os
import time

import utils.box_utils as box_utils
import utils.pc_utils as pc_utils
import utils.rotation as rotation
import utils.taxonomy as taxonomy
from utils.tenFpsDataLoader import TenFpsDataLoader, extract_gt
import utils.visual_utils as visual_utils

dc = taxonomy.ARKitDatasetConfig()

def accumulate_wrapper(loader, grid_size=0.05):
    """
    Args:
        loader: TenFpsDataLoader
    Returns:
        world_pc: (N, 3)
            xyz in world coordinate system
        world_sem: (N, d)
            semantic for each point
        grid_size: float
            keep only one point in each (g_size, g_size, g_size) grid
    """
    world_pc, world_rgb, poses = np.zeros((0, 3)), np.zeros((0, 3)), []
    for i in range(len(loader)):
        frame = loader[i]
        print(f"{i}/{len(loader)}", frame["image_path"])
        image_path = frame["image_path"]
        pcd = frame["pcd"]  # in world coordinate
        pose = frame["pose"]
        rgb = frame["color"]

        world_pc = np.concatenate((world_pc, pcd), axis=0)
        world_rgb = np.concatenate((world_rgb, rgb), axis=0)

        choices = pc_utils.down_sample(world_pc, 0.05)
        world_pc = world_pc[choices]
        world_rgb = world_rgb[choices]

    return world_pc, world_rgb, poses


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
        help="gt path to annotation .json file",
    )
    parser.add_argument("--frame_rate", default=1, help="sampling rate of frames")
    parser.add_argument(
        "--output_dir", default="../sample_data/offline_prepared_data/", help="directory to save the data and annoation"
    )
    parser.add_argument("--vis", action="store_true")

    args = parser.parse_args()

    # step 0.1: get annotation first,
    # if skipped or no gt boxes, we will not bother calling further steps
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
        class_names=taxonomy.class_names,
        root_path=data_path,
    )

    # step 0.3: output folder, make dir
    output_data_dir = os.path.join(args.output_dir, "%s_data" % args.scene_id)
    output_label_dir = os.path.join(args.output_dir, "%s_label" % args.scene_id)
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    t = time.time()
    # step 1: accumulate points and get full-scan for each box
    world_pc, world_rgb, _ = accumulate_wrapper(loader)
    points = world_pc.astype(np.float32)
    # save results
    # ./output_dir/{scene_id}_data/xxx_pc.npy
    data_save_fn = "%s_pc.npy" % args.scene_id
    data_save_fn = os.path.join(output_data_dir, data_save_fn)
    np.save(data_save_fn, points)

    # 2. boxes (boxes_corners: np.array (n, 8, 3))
    boxes = box_utils.corners_to_boxes(boxes_corners)
    corners_recon = box_utils.boxes_to_corners_3d(boxes)
    # corners_recon = [(-1, item) for item in corners_recon]
    gt_labels = {
        "bboxes": boxes,
        "types": labels,
        "uids": uids,
        "pose": [],
    }

    # gt boxes
    # ./output_dir/{scene_id}_label/xxx_bbox.npy
    box_save_fn = "%s_bbox.npy" % args.scene_id
    box_save_fn = os.path.join(output_label_dir, box_save_fn)
    np.save(box_save_fn, gt_labels)

    # visualization (for sanity check)
    if args.vis:
        # visual sanity-check: points should match boxes
        visual_utils.visualize_o3d(points[:, :3], corners_recon)

    elapased = time.time() - t
    print("total time: %f sec" % elapased)
