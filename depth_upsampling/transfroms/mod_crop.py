import dataset_keys


class ModCrop:
    def __init__(self, modulo: int):
        super().__init__()
        self.modulo = modulo

    def __call__(self, sample):
        img = sample[dataset_keys.COLOR_IMG]
        depth = sample[dataset_keys.HIGH_RES_DEPTH_IMG]
        tmpsz = depth.shape
        sz = [tmpsz[1], tmpsz[2]]
        sz[0] -= sz[0] % self.modulo
        sz[1] -= sz[1] % self.modulo

        img = img[:, :sz[0], :sz[1]]
        depth = depth[:, :sz[0], :sz[1]]

        sample[dataset_keys.COLOR_IMG] = img
        sample[dataset_keys.HIGH_RES_DEPTH_IMG] = depth
        return sample
