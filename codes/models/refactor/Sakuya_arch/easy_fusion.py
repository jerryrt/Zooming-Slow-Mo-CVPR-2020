import torch.nn as nn
from torchvision.ops import Conv2dNormActivation
from .pyramid_fusion import PyramidFusion


class EasyFusion(nn.Module):
    def __init__(self, multiplier=64, groups=8):
        super(EasyFusion, self).__init__()
        common_kwargs = dict(kernel_size=3,
                             # stride=2,
                             padding=1,
                             bias=True,
                             norm_layer=None,
                             activation_layer=nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.modules = nn.ModuleList([
            nn.Sequential(Conv2dNormActivation(multiplier, multiplier, stride=2, **common_kwargs),
                          Conv2dNormActivation(multiplier, multiplier, stride=1, **common_kwargs)),
            nn.Sequential(Conv2dNormActivation(multiplier, multiplier, stride=2, **common_kwargs),
                          Conv2dNormActivation(multiplier, multiplier, stride=1, **common_kwargs))])

        self.pyramid_fusion = PyramidFusion(multiplier=multiplier, groups=groups)
        self.fusion = nn.Conv2d(2 * multiplier, multiplier, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, features_a, features_b):
        pyramid_a, pyramid_b = [features_a], [features_b]
        for module in self.modules:
            pyramid_a.append(module(pyramid_a[-1]))
            pyramid_b.append(module(pyramid_b[-1]))

        x = self.pyramid_fusion(pyramid_a, pyramid_b)
        return self.fusion(x)
