import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation, DeformConv2d


class PyramidFusion(nn.Module):
    def __init__(self, num_levels=3, multiplier=64, groups=8):
        super(PyramidFusion, self).__init__()
        common_kwargs = dict(kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=True,
                             norm_layer=None,
                             activation_layer=nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.inputs_fusions = nn.ModuleList([
            nn.Sequential(Conv2dNormActivation(multiplier * 2, multiplier, **common_kwargs),
                          Conv2dNormActivation(multiplier, multiplier, **common_kwargs))
        ])
        for _ in range(num_levels - 1):
            self.inputs_fusions.append(Conv2dNormActivation(multiplier * 2, multiplier, **common_kwargs))

        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.offsets_fusions = nn.ModuleList([
            nn.Sequential(Conv2dNormActivation(multiplier * 2, multiplier, **common_kwargs),
                          Conv2dNormActivation(multiplier, multiplier, **common_kwargs)) for _ in range(num_levels)])
        self.warp_maps = nn.ModuleList([
            DeformConv2d(multiplier,
                         multiplier,
                         kernel_size=3,
                         stride=1,
                         padding=1,
                         groups=groups) for _ in range(num_levels)])
        self.output_fusions = nn.ModuleList([
            Conv2dNormActivation(multiplier * 2, multiplier, **common_kwargs) for _ in range(num_levels)])

    def forward(self, pyramid_a, pyramid_b):
        residual_offsets = []
        for fusion, features in zip(self.inputs_fusions, zip(pyramid_a, pyramid_b)):
            x = torch.cat(features, dim=1)
            x = fusion(x)
            residual_offsets.append(x)

        offsets = residual_offsets[:1]
        for fusion, offset in zip(self.offsets_fusions, residual_offsets[1:]):
            x = self.upsample2x(offsets[-1])
            x = torch.cat((offset, x * 2), dim=1)
            x = fusion(x)
            offsets.append(x)

        pyramid = [warp(level, offsets) for warp, level, offsets in zip(self.warp_maps, pyramid_a, offsets)]

        x = pyramid[0]
        for fusion, level in zip(self.out_fusions, pyramid[1:]):
            x = self.upsample2x(x)
            x = torch.cat((level, x), dim=1)
            x = fusion(x)
        return x
