import torch.nn as nn


class ConvPReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DeconvPReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel,
                                       stride=stride, padding=padding, output_padding=stride - 1)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
