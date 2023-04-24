import torch
import torch.nn as nn
from .pyramid_fusion import PyramidFusion


class SymmetricPyramidFusion(nn.Module):
    def __init__(self, multiplier=64, groups=8, num_levels=3):
        super(SymmetricPyramidFusion, self).__init__()
        self.module_ab = PyramidFusion(multiplier, groups, num_levels)
        self.module_ba = PyramidFusion(multiplier, groups, num_levels)
        self.fusion = nn.Conv2d(2 * multiplier, multiplier, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, pyramid_a, pyramid_b):
        features_ab, features_ba = self.module_ab(pyramid_a, pyramid_b), self.module_ba(pyramid_b, pyramid_a)
        x = torch.cat((features_ab, features_ba), dim=1)
        return self.fusion(x)
