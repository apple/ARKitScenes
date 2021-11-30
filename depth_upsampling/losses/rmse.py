import torch
import torch.nn.functional as F

import dataset_keys


def rmse_loss(outputs, inputs):
    valid_mask = inputs[dataset_keys.VALID_MASK_IMG]
    gt_depth = inputs[dataset_keys.HIGH_RES_DEPTH_IMG]
    prediction = outputs[dataset_keys.PREDICTION_DEPTH_IMG]
    loss = F.mse_loss(prediction[valid_mask], gt_depth[valid_mask])
    loss = torch.sqrt(loss)
    return loss
