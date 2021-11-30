from collections import defaultdict

import numpy as np
import torch

import dataset_keys
import image_utils
from data_utils import image_hwc_to_chw, image_chw_to_hwc, batch_to_cuda

MAX_TENSORBOARD_FRAMES = 10
VMIN = 0
VMAX = 7


def compute_errors(gt, pred, valid_mask):
    l1 = np.abs((gt - pred) * valid_mask)
    l2 = l1 ** 2

    image_dim = (1, 2)
    denominator = np.count_nonzero(valid_mask, axis=image_dim)
    l1_mean = np.sum(l1, image_dim) / denominator
    rmse = np.sqrt(np.sum(l2, image_dim) / denominator)

    return dict(L1=l1_mean, RMSE=rmse)


def eval_log(step, model, dataloader, tensorboard_writer):
    model.eval()
    with torch.no_grad():
        metrics = defaultdict(list)
        images_added_to_tensorboard = 0
        total_samples = 0
        for i, input_batch in enumerate(dataloader):
            input_batch = batch_to_cuda(input_batch)
            output_batch = model(input_batch)
            rgb = input_batch[dataset_keys.COLOR_IMG].cpu().numpy()
            gt_depth = input_batch[dataset_keys.HIGH_RES_DEPTH_IMG].cpu().numpy().squeeze(1)
            depth_lowres = input_batch[dataset_keys.LOW_RES_DEPTH_IMG].cpu().numpy().squeeze(1)
            valid_mask = input_batch[dataset_keys.VALID_MASK_IMG].cpu().numpy().squeeze(1)
            pred_depth = output_batch[dataset_keys.PREDICTION_DEPTH_IMG].cpu().numpy().squeeze(1)

            batch_size = rgb.shape[0]
            total_samples += batch_size
            batch_metrics = compute_errors(gt_depth, pred_depth, valid_mask)
            for key in batch_metrics:
                metrics[key].append(batch_metrics[key])

            j = 0
            while j < batch_size and images_added_to_tensorboard <= MAX_TENSORBOARD_FRAMES:
                identifier = input_batch[dataset_keys.IDENTIFIER][j]
                image_list = [image_chw_to_hwc(rgb[j]),
                              image_utils.colorize(gt_depth[j], VMIN, VMAX),
                              image_utils.colorize(depth_lowres[j], VMIN, VMAX),
                              image_utils.colorize(pred_depth[j], VMIN, VMAX)]

                h, w = pred_depth[j].shape[:2]
                montage = image_utils.create_montage_image(image_list, (w, h), grid_shape=(4, 1))
                tensorboard_writer.add_image(f'{identifier}', image_hwc_to_chw(montage / 255),
                                             step)
                j += 1
                images_added_to_tensorboard += 1

        print(f'validation metrics')
        print(("{:>7}, " * len(metrics)).format(*metrics.keys()))
        for key in metrics:
            metrics[key] = np.concatenate(metrics[key])
            metric = np.sum(metrics[key]) / total_samples
            print('{:7.3f}, '.format(metric), end='')
            tensorboard_writer.add_scalar(key, metric, step)
        print()
    model.train()
