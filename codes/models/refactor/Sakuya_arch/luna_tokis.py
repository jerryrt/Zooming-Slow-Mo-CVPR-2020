import torch
import torch.nn as nn
from .resnet_based_generator import ResnetBasedGenerator as Generator
from .pyramid_fusion import PyramidFusion
from .resnet_based_pyramid_extractor import ResnetBasedPyramidExtractor as PyramidExtractor
from .bidirectional_block import BidirectionalBlock


class LunaTokis(nn.Module):
    def __init__(self, in_channels, out_channels, multiplier=64, groups=8, front_RBs=5, back_RBs=10):
        super(LunaTokis, self).__init__()
        self.feature_extractor = PyramidExtractor(in_channels, multiplier, n_blocks=front_RBs)
        self.pyramid_fusion = PyramidFusion(multiplier=multiplier, groups=groups)
        self.fusion = nn.Conv2d(2 * multiplier, multiplier, kernel_size=1, stride=1, bias=True)
        self.ConvBLSTM = BidirectionalBlock(in_channels=multiplier,
                                            out_channels=multiplier,
                                            kernel_size=3,
                                            num_layers=1,
                                            fusion_groups=8)
        self.decoder = Generator(multiplier,
                                 out_channels=out_channels,
                                 n_blocks=back_RBs,
                                 upscale_factor=2)

    def forward(self, x):
        pyramid_a = self.feature_extractor(x[0])
        out = [pyramid_a[0]]
        for t in range(1, len(x)):
            pyramid_b = self.feature_extractor(x[t])
            features = self.pyramid_fusion(pyramid_a, pyramid_b)
            features = self.fusion(features)
            out.append(features)
            out.append(pyramid_b[0])
            pyramid_a = pyramid_b
        x = torch.stack(out, dim=1)
        x = self.ConvBLSTM(x)
        return torch.stack([self.decoder(x[t]) for t in range(len(x))])
