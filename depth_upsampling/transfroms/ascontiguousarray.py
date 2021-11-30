import numpy as np


class AsContiguousArray:
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        for key in sample:
            if isinstance(sample[key], np.ndarray):
                sample[key] = np.ascontiguousarray(sample[key])
        return sample
