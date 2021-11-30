import random

import numpy as np

import dataset_keys


class RandomFilpLR:
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        if random.randint(0, 1):
            asset_keys = [dataset_keys.COLOR_IMG, dataset_keys.HIGH_RES_DEPTH_IMG, dataset_keys.LOW_RES_DEPTH_IMG]
            for key in asset_keys:
                sample[key] = np.flip(sample[key], 2)
        return sample
