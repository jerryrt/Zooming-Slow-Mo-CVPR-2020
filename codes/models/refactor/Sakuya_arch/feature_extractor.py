import torch.nn as nn
from codes.models.refactor.Sakuya_arch.residual_block_no_bn import ResidualBlockNoBN as Block
# from .residual_block_no_bn import ResidualBlockNoBN as Block
from torchvision.ops import Conv2dNormActivation


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
                                 activation_layer=nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            *[Block(multiplier) for _ in range(n_blocks)]
        )


if __name__ == "__main__":
    FeatureExtractor(3, 64, 5)