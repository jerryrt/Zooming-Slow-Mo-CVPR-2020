import torch.nn as nn
from .functional import LeakyReLU1EM1_
from torchvision.ops import Conv2dNormActivation


class PyramidExtractor(nn.Module):
    def __init__(self, multiplier):
        super(PyramidExtractor, self).__init__()
        conv2d_kwargs = dict(kernel_size=3, padding=1, bias=True)
        common_kwargs = dict(**conv2d_kwargs, norm_layer=None, activation_layer=LeakyReLU1EM1_)
        self.levels = nn.ModuleList([
            nn.Sequential(Conv2dNormActivation(multiplier, multiplier, stride=2, **common_kwargs),
                          Conv2dNormActivation(multiplier, multiplier, stride=1, **common_kwargs)),
            nn.Sequential(Conv2dNormActivation(multiplier, multiplier, stride=2, **common_kwargs),
                          Conv2dNormActivation(multiplier, multiplier, stride=1, **common_kwargs))
        ])

    def forward(self, x):
        pyramid = [x]
        for module in self.levels:
            pyramid.append(module(pyramid[-1]))
        return pyramid
