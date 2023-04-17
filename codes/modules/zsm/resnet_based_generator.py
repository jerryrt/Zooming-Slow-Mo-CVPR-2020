import torch.nn as nn
from .functional import ResidualBlockNoBN as Block


class ResnetBasedGenerator(nn.Sequential):
    def __init__(self, multiplier, out_channels, n_blocks=10, upscale_factor=2):
        common_kwargs = dict(kernel_size=3, stride=1, padding=1, bias=True)
        super(ResnetBasedGenerator, self).__init__(
            *[Block(multiplier) for _ in range(n_blocks)],
            nn.Conv2d(multiplier, multiplier * (upscale_factor ** 2), **common_kwargs),
            nn.PixelShuffle(upscale_factor),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(multiplier, multiplier * (upscale_factor ** 2), **common_kwargs),
            nn.PixelShuffle(upscale_factor),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(multiplier, multiplier, **common_kwargs),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(multiplier, out_channels, **common_kwargs)
        )
