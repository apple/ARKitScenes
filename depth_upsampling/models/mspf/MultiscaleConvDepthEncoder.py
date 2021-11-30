import torch.nn.functional as torch_nn_func
from torch import nn

from models.mspf.blocks.multi_scale_depth import Conv2D


class MultiscaleConvDepthEncoder(nn.Module):
    """
    Inspired by: Multi-Scale Progressive Fusion Learning for Depth Map Super-Resolution
    https://arxiv.org/pdf/2011.11865v1.pdf
    """
    def __init__(self, upsample_factor):
        super().__init__()
        self.scale = int(upsample_factor)
        print("self.scale", self.scale)
        activation = "relu"
        batch_norm = None
        self.output_channels = [16, 32, 32, 64, 64, 128]

        self.conv_layers1 = nn.Sequential(
            Conv2D(
                1,
                self.output_channels[0],
                kernel_size=3,
                activation=activation,
                padding=1,
                batch_norm=batch_norm,
            ),
            Conv2D(
                self.output_channels[0],
                self.output_channels[0],
                kernel_size=3,
                activation=activation,
                padding=1,
                batch_norm=batch_norm,
            ))

        self.encoder_conv_blocks = []

        for i in range(1, 6):
            conv_block = nn.Sequential(
                Conv2D(
                    self.output_channels[i-1],
                    self.output_channels[i],
                    kernel_size=3,
                    activation=activation,
                    padding=1,
                    batch_norm=batch_norm,
                ),
                Conv2D(
                    self.output_channels[i],
                    self.output_channels[i],
                    kernel_size=2,
                    activation=activation,
                    stride=2,
                    padding=0,
                    batch_norm=batch_norm,
                ))

            setattr(self, f"conv_block_{i}", conv_block)
            self.encoder_conv_blocks.append(conv_block)

    def forward(self, depth):

        depth = torch_nn_func.interpolate(depth, scale_factor=self.scale, mode='bicubic', align_corners=True)
        features = self.conv_layers1(depth)

        skip_connections = {}
        skip_connections["x1"] = features

        for i in range(5):
            features = self.encoder_conv_blocks[i](features)
            skip_connections[f"x{(2)**(i+1)}"] = features

        return skip_connections




