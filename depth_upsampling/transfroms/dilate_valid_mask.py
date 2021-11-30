import torch

import dataset_keys


class DilateValidMask:
    def __init__(self, dilation_radius: int):
        self.dilation_radius = dilation_radius
        self.layer = torch.nn.MaxPool2d(dilation_radius * 2 + 1, stride=1, padding=dilation_radius)

    def __call__(self, batch):
        if dataset_keys.VALID_MASK_IMG in batch:
            batch[dataset_keys.VALID_MASK_IMG] = self.layer((~batch[dataset_keys.VALID_MASK_IMG]).float()) == 0
        return batch

