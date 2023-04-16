import torch.nn as nn
from .symmetric_pyramid_fusion import SymmetricPyramidFusion
from .pyramid_extractor import PyramidExtractor


class EasyFusion(nn.Module):
    def __init__(self, multiplier=64, groups=8):
        super(EasyFusion, self).__init__()
        self.pyramid_extractor = PyramidExtractor(multiplier=multiplier)
        self.pyramid_fusion = SymmetricPyramidFusion(multiplier=multiplier, groups=groups)

    def forward(self, features_a, features_b):
        pyramid_a, pyramid_b = self.pyramid_extractor(features_a), self.pyramid_extractor(features_b)
        return self.pyramid_fusion(pyramid_a[::-1], pyramid_b[::-1])
