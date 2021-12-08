# Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py

import numpy as np
import time

from .box_utils import box3d_iou


class APCalculator(object):
    """ Calculating Average Precision """

    def __init__(
        self,
        ap_iou_thresh=0.25,
        class2type_map=None,
    ):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: dict
                int: str
                e.g. {0: 'cabinet', 1: '', ...}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        n_class = len(self.class2type_map)
        self.class_names = []
        for i in range(n_class):
            self.class_names.append(self.class2type_map[i])

        self.reset()

    def step(
        self,
        batch_pred,
        batch_gt,
    ):
        """Accumulate one batch of prediction and groundtruth.
            self.gt_map_cls[idx]: a list of (int, np.array (8, 3))
            self.gt_map_cls[idx]: a list of (int, np.array (8, 3))

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred)
        assert bsize == len(batch_gt)
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        """Use accumulated predictions and groundtruths to compute Average Precision.
        # 1. calculate by-category precision, recall with multiprocessing mode
            each process for each category.

        Args:

        Returns:
            ret_dict: dict
                mAP: float (average AP over all categories)
                AR: float (average recall over all categories)
        """
        pred = {}
        gt = {}

        for img_id in pred_all.keys():
            for classname, bbox, score in pred_all[img_id]:
                if classname not in pred: pred[classname] = {}
                if img_id not in pred[classname]:
                    pred[classname][img_id] = []
                if classname not in gt: gt[classname] = {}
                if img_id not in gt[classname]:
                    gt[classname][img_id] = []
                pred[classname][img_id].append((bbox,score))
        
        for img_id in gt_all.keys():
            for classname, bbox in gt_all[img_id]:
                if classname not in gt: gt[classname] = {}
                if img_id not in gt[classname]:
                    gt[classname][img_id] = []
                gt[classname][img_id].append(bbox)
        
        rec = {}
        prec = {}
        ap = {}
        for classname in gt.keys():
            print('Computing AP for class: ', classname)
            rec[classname], prec[classname], ap[classname] = eval_det_cls(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
            print(classname, ap[classname])

        ret_dict = {} 
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision'%(clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall'%(clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall'%(clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0


def eval_det_cls(
    pred,
    gt,
    ovthresh=0.25,
    use_07_metric=False,
    get_iou_func=box3d_iou,
    classname="",
):
    """Generic functions to compute precision/recall for object detection
    for a single class.
    Args:
        pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
        gt: map of {img_id: [bbox]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if True use VOC07 11 point method
        get_iou_func: function handle for get_iou_func(box1, box2)
        classname: int

    Returns:
        rec: numpy array of length nd
        prec: numpy array of length nd
        ap: scalar, average precision
        confidence: numpy array, used to find precision and recall in offline processing given specific conf_threshold
    """
    tt = time.time()

    msg = "compute pr for single class of {}".format(classname)
    print(msg)

    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(
            bbox
        )  # total number of gt boxes. This is max number of possible correct predictions
        class_recs[img_id] = {"bbox": bbox, "det": det}
    # pad empty list to all other imgids

    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {"bbox": np.array([]), "det": []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    # num of predicted box instances. say 100 boxes, and their img_ids (may contain a lot of duplicates, just appended)
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):  # global all img_ids
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_func(bb, BBGT[j, ...])
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        # check where 0.05 box confidence is used, for such low thr, fn is our hard examples to collect
        # if high threshold like 0.9, (fp is hard negative) we would be collecting hard FPs, this is because when we
        # set a high conf_threshold of 0.9, only those predicted boxes with very high confidence gets to be evaluated
        # so they are very likely to be TPs, in such scenario, if there is still FP, this means we encounter a hard FP
        # or hard negative example

        # confidence threshold changed from 0.05 to 0.9 to get false negatives
        # text file: 14 X 2 numbers of text files
        # for each category, we have a FP list, and a FN list
        # for each table list, we have full file path of 34578274_box_78.npy / npz, space, number of FPs,
        # for each table list, ... FNs

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                # even though the IoU is more than IoU threshold, but there is already a box earlier with higher
                # confidence that marked this gt box as having a TP detection, as ft[d] set to 1
                # img_id, no. of fp += 1
                fp[d] = 1.0
                # save_num_of_fp_per_img_id(classname, image_ids[d])
        else:
            # no gt box has IoU more than threshold
            fp[d] = 1.0
            # img_id, no. of fp += 1
            # save_num_of_fp_per_img_id(classname, image_ids[d])

    # compute precision recall
    tp_per_instance = tp.copy()
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos + 1e-4)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    msg = "----------------------- time for evaluating model: {} seconds".format(
        int(time.time() - tt)
    )
    print(msg)

    return rec, prec, ap

def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
