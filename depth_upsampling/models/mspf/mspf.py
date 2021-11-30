from torch import nn

import dataset_keys
from models.mspf.MultiScaleDepthSR import MultiscaleDepthDecoder
from models.mspf.densenet import DenseNet121


class MSPF(nn.Module):
    """
    Inspired by: Multi-Scale Progressive Fusion Learning for Depth Map Super-Resolution
    https://arxiv.org/pdf/2011.11865v1.pdf

    variables:
        decoder_channel_output_scales: available scale factors for reducing channels
        in decoder based on encoder input channels.
    params:
        decoder_channel_scale: used to control decoder size

    """

    decoder_channel_output_scales = [1, 2, 4, 8, 16]

    def __init__(self, upsample_factor, decoder_channel_scale=2):
        super(MSPF, self).__init__()
        self.encoder = DenseNet121()
        assert decoder_channel_scale in self.decoder_channel_output_scales, \
            f"decoder scale factor not supported {decoder_channel_scale} - supported {self.decoder_channel_output_scales}"
        input_channels = self.encoder.skip_out_channels[::-1]
        output_channels = [int(ch/decoder_channel_scale) for ch in self.encoder.skip_out_channels[::-1]]

        self.decoder = MultiscaleDepthDecoder(input_channels, output_channels, upsample_factor)

    def forward(self, batch):
        rgb = batch[dataset_keys.COLOR_IMG]
        rgb = (rgb / 255.0) - 0.5
        depth = batch[dataset_keys.LOW_RES_DEPTH_IMG]
        skip_features = self.encoder(rgb)
        output = self.decoder(depth, skip_features)
        return {dataset_keys.PREDICTION_DEPTH_IMG: output}
