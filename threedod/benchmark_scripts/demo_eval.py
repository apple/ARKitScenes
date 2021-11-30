import numpy as np
import os
import sys

from utils.box_utils import boxes_to_corners_3d
from utils.eval_utils import eval_det_cls


if __name__ == "__main__":
    boxes_gt = np.array(
        [[0., 0., 0., 1., 1., 1., 0.2]]
    )
    corners_gt = boxes_to_corners_3d(boxes_gt)

    boxes_pred = np.array(
        [[0.1, 0.1, -0.1, 1.1, .8, .9, 0.3],
         [0.2, 0.1, 0.3, 1., 1., 1., -0.1],
         [2.2, 1.1, 3.3, 1., 1., 1., -0.1]
        ]
    )
    conf = [0.7, 0.3, 0.8]
    corners_pred = boxes_to_corners_3d(boxes_pred)

    pred = {0: []}
    gt = {0: []}

    for i in range(corners_gt.shape[0]):
        gt[0].append(corners_gt[i])

    for i in range(corners_pred.shape[0]):
        pred[0].append((corners_pred[i], conf[i]))

    rec, prec, ap = eval_det_cls(pred, gt, classname="cabinet")
    print(rec, prec, ap)