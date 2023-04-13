import torch.nn as nn
from .resnet_based_generator import ResnetBasedGenerator as Generator
from .feature_extractor import FeatureExtractor
from ..bidirectional_module import BidirectionalModule
from ..unpack_module import UnpackModule

from .easy_fusion import EasyFusion
from .main_block import MainBlock


class ZSM(nn.Module):
    def __init__(self, in_channels, out_channels, multiplier=64, groups=8, front_RBs=5, back_RBs=10):
        super(ZSM, self).__init__()
        self.feature_extractor = FeatureExtractor(in_channels, multiplier, n_blocks=front_RBs)
        self.fusion = EasyFusion(multiplier=64, groups=8)
        self.main_module = BidirectionalModule(UnpackModule(UnpackModule(MainBlock(in_channels=multiplier,
                                                                                   out_channels=multiplier,
                                                                                   kernel_size=3,
                                                                                   num_layers=1,
                                                                                   fusion_groups=8), idx=1), idx=0))
        self.decoder = Generator(multiplier,
                                 out_channels=out_channels,
                                 n_blocks=back_RBs,
                                 upscale_factor=2)

    def forward(self, x):
        features = self.feature_extractor(x[0])
        out = [features]
        for t in range(1, len(x)):
            features_t = self.feature_extractor(x[t])
            out.append(self.fusion(features_a=out[-1], features_b=features_t))
            out.append(features_t)

        #x = torch.stack(out, dim=1)
        x = out
        x = self.main_module(x)
#        return torch.stack([self.decoder(x[t]) for t in range(len(x))])
        return [self.decoder(x[t]) for t in range(len(x))]


def zsm():
    pass
