import numpy as np
import torch
from torch import nn

from .msg.msg import MSGNet
from .mspf.mspf import MSPF


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def get_network(network, upsampling_factor):
    # Create model
    if network == 'MSG':
        model = MSGNet(upsampling_factor)
    elif network == 'MSPF':
        model = MSPF(upsampling_factor)
        model.decoder.apply(weights_init_xavier)
    else:
        raise ValueError(f'No such network ({network})')

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))
    return model
