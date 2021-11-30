from typing import List

import torch
from torch import nn

from models.mspf.MultiscaleConvDepthEncoder import MultiscaleConvDepthEncoder
from models.mspf.blocks.multi_scale_depth import Upsample2D, Conv2D

"""
- create conv for matching output channels
- skip convs to reduce channels
"""


class MultiscaleDepthDecoder(nn.Module):
    """
    Inspired by: Multi-Scale Progressive Fusion Learning for Depth Map Super-Resolution
    https://arxiv.org/pdf/2011.11865v1.pdf
    """
    def __init__(
        self,
        input_channels: List[int],
        output_channels: List[int],
        upsample_factor: int
    ):
        super().__init__()

        activation = "relu"
        batch_norm = None
        self.scales = ["x32", "x16", "x8", "x4", "x2", "x1"]

        self.depth_encoder = MultiscaleConvDepthEncoder(upsample_factor)
        depth_output_channels = self.depth_encoder.output_channels[::-1]
        rgb_output_channels = input_channels

        self.upsample_blocks = {}
        for i in range(len(self.scales)-1):
            ch_input = rgb_output_channels[i] + \
                       depth_output_channels[i] + \
                       (output_channels[i-1] if i > 0 else 0)
            conv_layers = nn.Sequential(
                Conv2D(
                    ch_input,
                    output_channels[i],
                    kernel_size=3,
                    activation=activation,
                    padding=1,
                    batch_norm=batch_norm,
                ),
                Conv2D(
                    output_channels[i],
                    output_channels[i],
                    kernel_size=3,
                    activation=activation,
                    padding=1,
                    batch_norm=batch_norm,
                ))

            upsample = Upsample2D(
                output_channels[i],
                output_channels[i],
            )
            setattr(self, f"conv_layers_{self.scales[i]}", conv_layers)
            setattr(self, f"upsample_{self.scales[i]}", upsample)
            self.upsample_blocks[self.scales[i]] = (conv_layers, upsample)

        ch_input = 3 + \
                   depth_output_channels[-1] + \
                   (output_channels[i - 1] if i > 0 else 0)

        conv_layers = nn.Sequential(
            Conv2D(
                ch_input,
                output_channels[i],
                kernel_size=3,
                activation=activation,
                padding=1,
                batch_norm=batch_norm,
            ),
            Conv2D(
                output_channels[i],
                1,
                kernel_size=3,
                activation=None,
                padding=1,
                batch_norm=batch_norm,
            ))
        setattr(self, f"conv_layers_x1", conv_layers)
        self.upsample_blocks['x1'] = (conv_layers, None)

    def forward(
        self, depth: dict, rgb_skip_connections: dict
    ):

        depth_skip_connections = self.depth_encoder(depth)

        fusion_features = None
        for scale, (conv_layers, upsample) in self.upsample_blocks.items():

            if fusion_features is None:
                fusion_features = torch.cat((rgb_skip_connections[scale], depth_skip_connections[scale]), 1)
            else:
                fusion_features = torch.cat((rgb_skip_connections[scale], depth_skip_connections[scale], fusion_features), 1)

            fusion_features = conv_layers(fusion_features)

            if upsample is not None:
                fusion_features = upsample(fusion_features)

        depth = fusion_features

        return depth
