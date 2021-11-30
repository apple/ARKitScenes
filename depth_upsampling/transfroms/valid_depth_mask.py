import dataset_keys


class ValidDepthMask:
    def __init__(self, gt_low_limit: float = None, gt_high_limit: float = None):
        assert gt_low_limit is None or gt_low_limit > 0, f'gt_low_limit must be greater than 0'
        assert gt_high_limit is None or gt_high_limit > 0, f'gt_high_limit must be greater than 0'
        self.gt_low_limit = gt_low_limit
        self.gt_high_limit = gt_high_limit

    def __call__(self, sample):
        if dataset_keys.VALID_MASK_IMG in sample:
            valid_mask = sample[dataset_keys.VALID_MASK_IMG]
        else:
            valid_mask = sample[dataset_keys.HIGH_RES_DEPTH_IMG] != 0
        if self.gt_low_limit is not None:
            valid_mask = valid_mask & (sample[dataset_keys.HIGH_RES_DEPTH_IMG] > self.gt_low_limit)
        if self.gt_high_limit is not None:
            valid_mask = valid_mask & (sample[dataset_keys.HIGH_RES_DEPTH_IMG] < self.gt_high_limit)
        sample[dataset_keys.VALID_MASK_IMG] = valid_mask
        return sample
