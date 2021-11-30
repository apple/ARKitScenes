import random

import dataset_keys


class RandomCrop:
    def __init__(self, height: int, width: int, upsample_factor: int = None):
        super().__init__()
        self.height = height
        self.width = width
        self.upsample_factor = upsample_factor

    def __call__(self, sample):
        low_res = sample[dataset_keys.LOW_RES_DEPTH_IMG].shape
        low_res_patch_width = int(self.width / self.upsample_factor)
        low_res_patch_height = int(self.height / self.upsample_factor)
        x = random.randint(0, low_res[2] - low_res_patch_width)
        y = random.randint(0, low_res[1] - low_res_patch_height)

        # crop low resolution depth image
        img = sample[dataset_keys.LOW_RES_DEPTH_IMG]
        img = img[:, y:y + low_res_patch_height, x:x + low_res_patch_width]
        sample[dataset_keys.LOW_RES_DEPTH_IMG] = img

        # crop remaining
        y *= self.upsample_factor
        x *= self.upsample_factor
        for key in [dataset_keys.COLOR_IMG, dataset_keys.HIGH_RES_DEPTH_IMG]:
            sample[key] = sample[key][:, y:y + self.height, x:x + self.width]
        return sample
