import torch

import dataset_keys


def div_by_mask_sum(loss: torch.Tensor, mask_sum: torch.Tensor):
    return loss / torch.max(mask_sum, torch.ones_like(mask_sum))


class SafeTorchLog(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        input_abs = torch.abs(input) + 1e-9
        ctx.save_for_backward(input_abs)

        return torch.log(input_abs)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (input_abs,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        return grad_input * (1.0 / input_abs) / 2.302585093  # ln(10)


safe_torch_log = SafeTorchLog.apply


def create_gradient_log_loss(log_prediction_d, mask, log_gt):

    # compute log difference
    log_d_diff = log_prediction_d - log_gt
    log_d_diff = torch.mul(log_d_diff, mask)

    # compute vertical gradient
    v_gradient = torch.abs(log_d_diff[:, :, 2:, :] - log_d_diff[:, :, :-2, :])
    v_mask = torch.mul(mask[:, :, 2:, :], mask[:, :, :-2, :])
    v_gradient = torch.mul(v_gradient, v_mask)

    # compute horizontal gradient
    h_gradient = torch.abs(log_d_diff[:, :, :, 2:] - log_d_diff[:, :, :, :-2])
    h_mask = torch.mul(mask[:, :, :, 2:], mask[:, :, :, :-2])
    h_gradient = torch.mul(h_gradient, h_mask)

    # sum up gradients
    grad_loss = torch.sum(h_gradient, dim=[1, 2, 3]) + torch.sum(v_gradient, dim=[1, 2, 3])
    num_valid_pixels = torch.sum(mask, dim=[1, 2, 3])
    grad_loss = div_by_mask_sum(grad_loss, num_valid_pixels)

    return grad_loss


def create_gradient_log_loss_4_scales(log_prediction, log_ground_truth, mask):
    log_prediction_d = log_prediction
    log_gt = log_ground_truth
    mask = mask

    log_prediction_d_scale_1 = log_prediction_d[:, :, ::2, ::2]
    log_prediction_d_scale_2 = log_prediction_d_scale_1[:, :, ::2, ::2]
    log_prediction_d_scale_3 = log_prediction_d_scale_2[:, :, ::2, ::2]

    mask_scale_1 = mask[:, :, ::2, ::2]
    mask_scale_2 = mask_scale_1[:, :, ::2, ::2]
    mask_scale_3 = mask_scale_2[:, :, ::2, ::2]

    log_gt_scale_1 = log_gt[:, :, ::2, ::2]
    log_gt_scale_2 = log_gt_scale_1[:, :, ::2, ::2]
    log_gt_scale_3 = log_gt_scale_2[:, :, ::2, ::2]

    gradient_loss_scale_0 = create_gradient_log_loss(log_prediction_d, mask, log_gt)

    gradient_loss_scale_1 = create_gradient_log_loss(
        log_prediction_d_scale_1, mask_scale_1, log_gt_scale_1
    )

    gradient_loss_scale_2 = create_gradient_log_loss(
        log_prediction_d_scale_2, mask_scale_2, log_gt_scale_2
    )

    gradient_loss_scale_3 = create_gradient_log_loss(
        log_prediction_d_scale_3, mask_scale_3, log_gt_scale_3
    )

    gradient_loss_4_scales = (
        gradient_loss_scale_0 + gradient_loss_scale_1 + gradient_loss_scale_2 + gradient_loss_scale_3
    )

    return gradient_loss_4_scales


def gradient_loss(outputs, inputs):
    valid_mask = inputs[dataset_keys.VALID_MASK_IMG]
    gt_depth = inputs[dataset_keys.HIGH_RES_DEPTH_IMG]
    prediction = outputs[dataset_keys.PREDICTION_DEPTH_IMG]

    log_prediction = safe_torch_log(prediction)
    log_gt = safe_torch_log(gt_depth)
    loss = create_gradient_log_loss_4_scales(log_prediction, log_gt, valid_mask)
    loss = torch.mean(loss)
    return loss
