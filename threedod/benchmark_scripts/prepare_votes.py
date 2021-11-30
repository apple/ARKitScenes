import argparse
import numpy as np
import os
import sys
import time

from utils.box_utils import boxes_to_corners_3d, points_in_boxes
from utils.visual_utils import visualize_o3d


def get_votes(points, gt_boxes):
    """
    Args:
        points: (N, 3)
        boxes: (m, 8, 3)
    Returns:
        votes: (N, 4)
    """
    n_point = points.shape[0]
    point_votes = np.zeros((n_point, 4)).astype(np.float32)
    for obj_id in range(gt_boxes.shape[0]):
        tmp_box3d = np.expand_dims(gt_boxes[obj_id], 0)  # (8, 3)
        # (n_point, 1)
        mask_pts = points_in_boxes(points[:, :3], tmp_box3d)
        mask_pts = mask_pts.reshape((-1,))
        point_votes[mask_pts, 0] = 1.0
        obj_center = np.mean(tmp_box3d, axis=1)  # (1, 3)

        # get votes
        pc_roi = points[mask_pts, :3]
        tmp_votes = obj_center - pc_roi
        point_votes[mask_pts, 1:4] = tmp_votes
    return point_votes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="../../offline_prepared_data/40753679_data/40753679_pc.npy",
    )
    parser.add_argument(
        "--label_path",
        default="../../offline_prepared_data/40753679_label/40753679_bbox.npy",
    )
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    pc = np.load(args.data_path)
    label = np.load(args.label_path, allow_pickle=True).item()
    boxes = label["bboxes"]
    # make boxes a little bit larger to include more points
    boxes[:, 3:6] += 0.01
    gt_boxes = boxes_to_corners_3d(boxes)
    votes = get_votes(pc, gt_boxes)

    ## visualize votes
    if args.vis:
        votes_mask = votes[:, 0] > 0.
        votes = votes[votes_mask, 1:]

        pc = pc[votes_mask, :]
        center = pc + votes

        boxes_cls = []
        for i in range(boxes.shape[0]):
            # give a random -1 as class
            boxes_cls.append((-1, gt_boxes[i, :, :]))

        visualize_o3d(center, boxes_cls)
