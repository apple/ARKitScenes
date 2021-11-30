import numpy as np
import torch


def expand_channel_dim(img):
    """
        expand image dimension to add a channel dimension
    """
    return np.expand_dims(img, 0)


def image_hwc_to_chw(img):
    """
        transpose the image from height, width, channel -> channel, height, width
        (pytorch format)
    """
    return img.transpose((2, 0, 1))


def image_chw_to_hwc(img):
    """
        revert image_hwc_to_chw function
    """
    return img.transpose((1, 2, 0))


def batch_to_cuda(batch):
    if torch.cuda.is_available():
        for k in batch:
            if k != 'identifier':
                batch[k] = batch[k].cuda(non_blocking=True)
    return batch
