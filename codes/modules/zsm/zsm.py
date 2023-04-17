import torch.nn as nn
from .resnet_based_generator import ResnetBasedGenerator as Generator
from .feature_extractor import FeatureExtractor
from ..bidirectional_module import BidirectionalModule
from ..unpack import Unpack

from .easy_fusion import EasyFusion
from .main_block import MainBlock
from .functional import LeakyReLU1EM1, ResidualBlockNoBN as Block
from torchvision.ops import Conv2dNormActivation


class ZSM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multiplier=64,
                 groups=8,
                 num_front_block=5,
                 num_temporal_blocks=1,
                 num_generator_blocks=40,
                 upscale_factor=2):
        super(ZSM, self).__init__()
        # self.feature_extractor = FeatureExtractor(in_channels, multiplier, n_blocks=num_front_block)

        conv2d_kwargs = dict(kernel_size=3, padding=1, bias=True)
        common_kwargs = dict(**conv2d_kwargs, norm_layer=None, activation_layer=LeakyReLU1EM1)
        self.feature_extractor = nn.Sequential(Conv2dNormActivation(in_channels, multiplier, **common_kwargs),
                                               *[Block(multiplier) for _ in range(num_front_block)])

        self.fusion = EasyFusion(multiplier=64, groups=8)
        directional_block = nn.Sequential(MainBlock(in_channels=multiplier,
                                                    out_channels=multiplier,
                                                    kernel_size=3,
                                                    padding=1,
                                                    num_layers=num_temporal_blocks,
                                                    fusion_groups=groups,
                                                    return_all_layers=False),
                                          Unpack(indexes=(1, 0, 0)))
        self.main_module = BidirectionalModule(directional_block)
        self.output_fusion = nn.Conv2d(2 * multiplier, multiplier, kernel_size=1, stride=1, bias=True)

        # self.decoder = Generator(multiplier, out_channels, n_blocks=num_generator_blocks, upscale_factor=2)
        self.decoder = nn.Sequential(*[Block(multiplier) for _ in range(num_generator_blocks)],
                                     *[nn.Sequential(nn.Conv2d(multiplier, multiplier * 4, **conv2d_kwargs),
                                                     nn.PixelShuffle(2),
                                                     nn.LeakyReLU(negative_slope=0.1, inplace=True))
                                       for _ in range(upscale_factor)],
                                     Conv2dNormActivation(multiplier, multiplier, **common_kwargs),
                                     nn.Conv2d(multiplier, out_channels, **conv2d_kwargs))

    def forward(self, x):
        features = self.feature_extractor(x[0])
        out = [features]
        for t in range(1, len(x)):
            features_t = self.feature_extractor(x[t])
            out.append(self.fusion(features_a=out[-1], features_b=features_t))
            out.append(features_t)
        # x = torch.stack(out, dim=1)
        x = out
        x = self.main_module(x)
        #        return torch.stack([self.decoder(x[t]) for t in range(len(x))])
        return [self.decoder(self.output_fusion(x[t])) for t in range(len(x))]


def zsm():
    pass
