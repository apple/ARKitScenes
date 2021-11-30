import numpy as np
import torch.utils.data


class MultiEpochSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly over multiple epochs

    Arguments:
        data_source (Dataset): dataset to sample from
        num_iter (int) : Number of times to loop over the dataset
        start_itr (int) : which iteration to begin from
    """

    def __init__(self, data_source, num_iter, start_itr=0, batch_size=128):
        super().__init__(data_source)
        self.data_source = data_source
        self.dataset_size = len(self.data_source)
        self.num_iter = num_iter
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.num_epochs = int(np.ceil((self.num_iter * self.batch_size) / float(self.dataset_size)))

        if not isinstance(self.dataset_size, int) or self.dataset_size <= 0:
            raise ValueError("dataset size should be a positive integeral "
                             "value, but got dataset_size={}".format(self.dataset_size))

    def __iter__(self):
        n = self.dataset_size
        # Determine number of epochs
        num_epochs = int(np.ceil(((self.num_iter - self.start_itr) * self.batch_size) / float(n)))
        out = np.concatenate([np.random.permutation(n) for epoch in range(self.num_epochs)])[-num_epochs * n: self.num_iter * self.batch_size]
        out = out[(self.start_itr * self.batch_size % n):]
        return iter(out)

    def __len__(self):
        return (self.num_iter - self.start_itr) * self.batch_size
