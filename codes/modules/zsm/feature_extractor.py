import torch.nn as nn
from torchvision.ops import Conv2dNormActivation
from .functional import LeakyReLU1EM1, ResidualBlockNoBN as Block


#TODO: maybe move back into zsm
class FeatureExtractor(nn.Sequential):
    def __init__(self, in_channels, multiplier, n_blocks):
        super(FeatureExtractor, self).__init__(
            Conv2dNormActivation(in_channels,
                                 multiplier,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=True,
                                 norm_layer=None,
                                 activation_layer=LeakyReLU1EM1),
            *[Block(multiplier) for _ in range(n_blocks)]
        )
