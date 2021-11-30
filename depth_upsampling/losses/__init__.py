from .gradient_loss import gradient_loss
from .l1_loss import l1_loss
from .rmse import rmse_loss


def mspf_loss(output_batch, input_batch):
    return l1_loss(output_batch, input_batch) + 2 * gradient_loss(output_batch, input_batch)


def msg_loss(output_batch, input_batch):
    return rmse_loss(output_batch, input_batch)


def get_loss(network):
    if network == 'MSG':
        loss = msg_loss
    elif network == 'MSPF':
        loss = mspf_loss
    else:
        raise ValueError(f'No such network ({network})')
    return loss
