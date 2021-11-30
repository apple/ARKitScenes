import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .blocks import ConvPReLu, DeconvPReLu
import dataset_keys


class MSGNet(nn.Module):
    """
    Inspired by: Depth Map Super-Resolution by Deep Multi-Scale Guidance
    http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2016_depth.pdf
    """
    def __init__(self, upsampling_factor):
        super().__init__()
        # initialize indexes for layers
        self.upsampling_factor = upsampling_factor
        m = int(np.log2(upsampling_factor))

        # RGB-branch
        self.rgb_encoder1 = nn.Sequential(ConvPReLu(3, 49, 7, stride=1, padding=3),
                                          ConvPReLu(49, 32))
        self.rgb_encoder_blocks = nn.ModuleList()
        for i in range(m-1):
            self.rgb_encoder_blocks.append(nn.Sequential(ConvPReLu(32, 32),
                                                         nn.MaxPool2d(3, 2, padding=1)))

        # D-branch
        self.depth_decoder1 = nn.Sequential(ConvPReLu(1, 64, 5, stride=1, padding=2),
                                            DeconvPReLu(64, 32, 5, stride=2, padding=2))
        self.depth_decoder_blocks = nn.ModuleList()
        for i in range(m-1):
            self.depth_decoder_blocks.append(nn.Sequential(ConvPReLu(64, 32, 5, stride=1, padding=2),
                                                           ConvPReLu(32, 32, 5, stride=1, padding=2),
                                                           DeconvPReLu(32, 32, 5, stride=2, padding=2)))

        self.depth_decoder_n = nn.Sequential(ConvPReLu(64, 32, 5, stride=1, padding=2),
                                             ConvPReLu(32, 32, 5, stride=1, padding=2),
                                             ConvPReLu(32, 32, 5, stride=1, padding=2),
                                             ConvPReLu(32, 1, 5, stride=1, padding=2))

    def forward(self, batch):
        rgb_img = batch[dataset_keys.COLOR_IMG] / 255
        low_res_depth = batch[dataset_keys.LOW_RES_DEPTH_IMG]
        min_d = low_res_depth.amin((1, 2, 3), keepdim=True)
        max_d = low_res_depth.amax((1, 2, 3), keepdim=True)
        low_res_depth_norm = (low_res_depth - min_d) / ((max_d - min_d) + 1e-8)
        low_res_upsampled = F.interpolate(low_res_depth_norm, rgb_img.shape[2:], mode='bicubic')

        rgb_features = [self.rgb_encoder1(rgb_img), ]
        for block in self.rgb_encoder_blocks:
            rgb_features.append(block(rgb_features[-1]))

        rec = self.depth_decoder1(low_res_depth_norm)
        for i, block in enumerate(self.depth_decoder_blocks):
            rec = torch.cat((rec, rgb_features[-(i + 1)]), 1)
            rec = block(rec)
        rec = torch.cat((rec, rgb_features[0]), 1)
        rec = self.depth_decoder_n(rec)

        output = (low_res_upsampled + rec) * (max_d - min_d) + min_d
        return {dataset_keys.PREDICTION_DEPTH_IMG: output}
