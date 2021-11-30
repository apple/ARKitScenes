import torch

import dataset_keys

eps = 1e-6


def l1_loss(outputs, inputs):
    valid_mask = inputs[dataset_keys.VALID_MASK_IMG]
    gt_depth = inputs[dataset_keys.HIGH_RES_DEPTH_IMG]
    prediction = outputs[dataset_keys.PREDICTION_DEPTH_IMG]

    error_image = torch.abs(prediction - gt_depth) * valid_mask
    sum_loss = torch.sum(error_image, dim=[1, 2, 3])
    num_valid_pixels = torch.sum(valid_mask, dim=[1, 2, 3])
    loss = sum_loss / torch.max(num_valid_pixels, torch.ones_like(num_valid_pixels) * eps)
    return torch.mean(loss)
