import torch.nn as nn
from ..bidirectional_module import BidirectionalModule
from ..unpack import Unpack

from .easy_fusion import EasyFusion
from .main_block import MainBlock
from .functional import LeakyReLU1EM1_, ResidualBlockNoBN as Block
from torchvision.ops import Conv2dNormActivation


class ZSM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multiplier=64,
                 groups=8,
                 num_extractor_blocks=5,
                 num_temporal_blocks=1,
                 num_generator_blocks=40,
                 num_upscales=2):
        super(ZSM, self).__init__()
        conv2d_kwargs = dict(kernel_size=3, padding=1, bias=True)
        common_kwargs = dict(**conv2d_kwargs, norm_layer=None, activation_layer=LeakyReLU1EM1_)

        self.feature_extractor = nn.Sequential(Conv2dNormActivation(in_channels, multiplier, **common_kwargs),
                                               *[Block(multiplier) for _ in range(num_extractor_blocks)])

        self.fusion = EasyFusion(multiplier=multiplier, groups=groups)
        directional_block = nn.Sequential(MainBlock(in_channels=multiplier,
                                                    out_channels=multiplier,
                                                    kernel_size=3,
                                                    padding=1,
                                                    num_layers=num_temporal_blocks,
                                                    fusion_groups=groups),
                                          Unpack(indexes=(1, 0, 0)))
        self.main_module = BidirectionalModule(directional_block)
        self.output_fusion = nn.Conv2d(2 * multiplier, multiplier, kernel_size=1, stride=1, bias=True)

        self.decoder = nn.Sequential(*[Block(multiplier) for _ in range(num_generator_blocks)],
                                     *[nn.Sequential(nn.Conv2d(multiplier, multiplier * 4, **conv2d_kwargs),
                                                     nn.PixelShuffle(2),
                                                     LeakyReLU1EM1_()) for _ in range(num_upscales)],
                                     Conv2dNormActivation(multiplier, multiplier, **common_kwargs),
                                     nn.Conv2d(multiplier, out_channels, **conv2d_kwargs))
        self.upscale_factor = 2 ** num_upscales

    def forward(self, x):
        features = self.feature_extractor(x[0])
        out = [features]
        for t in range(1, len(x)):
            features_t = self.feature_extractor(x[t])
            out.append(self.fusion(features_a=out[-1], features_b=features_t))
            out.append(features_t)
        x = self.main_module(out)
        return [self.decoder(self.output_fusion(x[t])) for t in range(len(x))]

    def extract_features_pyramid(self, x):
        pass

    def run_on_features(self, features):
        pass


def zsm():
    pass
