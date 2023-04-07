import torch.nn as nn
from .residual_block_no_bn import ResidualBlockNoBN as Block
from torchvision.ops import Conv2dNormActivation


class ResnetBasedPyramidExtractor(nn.Module):
    def __init__(self, in_channels, multiplier, n_blocks):
        super(ResnetBasedPyramidExtractor, self).__init__()
        common_kwargs = dict(kernel_size=3,
                             padding=1,
                             bias=True,
                             norm_layer=None,
                             activation_layer=nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.levels = nn.ModuleList([
            nn.Sequential(Conv2dNormActivation(in_channels, multiplier, stride=1, **common_kwargs),
                          *[Block(multiplier) for _ in range(n_blocks)]),
            nn.Sequential(Conv2dNormActivation(multiplier, multiplier, stride=2, **common_kwargs),
                          Conv2dNormActivation(multiplier, multiplier, stride=1, **common_kwargs)),
            nn.Sequential(Conv2dNormActivation(multiplier, multiplier, stride=2, **common_kwargs),
                          Conv2dNormActivation(multiplier, multiplier, stride=1, **common_kwargs))])

    def forward(self, x):
        pyramid = []
        for module in self.encoder:
            x = module(x)
            pyramid.append(x)
        return pyramid
