import torch.nn as nn
from .pyramid_fusion import PyramidFusion
from .pyramid_extractor import PyramidExtractor


class EasyFusion(nn.Module):
    def __init__(self, multiplier=64, groups=8):
        super(EasyFusion, self).__init__()
        self.pyramid_extractor = PyramidExtractor(multiplier=multiplier)
        self.pyramid_fusion = PyramidFusion(multiplier=multiplier, groups=groups)
        self.fusion = nn.Conv2d(2 * multiplier, multiplier, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, features_a, features_b):
        pyramid_a, pyramid_b = self.pyramid_extractor(features_a), self.pyramid_extractor(features_b)
        x = self.pyramid_fusion(pyramid_a, pyramid_b)
        return self.fusion(x)
