import torch.nn.functional as torch_nn_func
from torch import nn


class Conv2D(nn.Module):
    """
    P = ((S-1)*W-S+F)/2, with F = filter size, S = stride

    """
    def __init__(self, in_channels, out_channels, bias=False, kernel_size=1, stride=1, padding=0, dilation=1, activation=None, batch_norm=None):
        super(Conv2D, self).__init__()

        self.activation = activation
        self.norm = batch_norm

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

        if self.norm is not None:
            self.norm = nn.BatchNorm2d

        if self.activation is not None:
            if self.activation == "relu":
                self.activation = nn.ReLU()
            else:
                raise Exception(f"activation {self.activation} not supported")

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(Upsample2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              bias=False, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.ratio = ratio

    def forward(self, x):
        up_x = torch_nn_func.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.relu(out)
        return out


