import torch
import torch.nn as nn
from .functional import LeakyReLU1EM1_
from .warp_net import WarpNet
from torchvision.ops import Conv2dNormActivation


class PyramidFusion(nn.Module):
    def __init__(self, multiplier=64, groups=8, num_levels=3):
        super(PyramidFusion, self).__init__()
        conv2d_kwargs = dict(kernel_size=3, stride=1, padding=1, bias=True)
        common_kwargs = dict(**conv2d_kwargs, norm_layer=None, activation_layer=LeakyReLU1EM1_)

        self.offsets_fusions = nn.ModuleList([
            nn.Sequential(Conv2dNormActivation(multiplier * 2, multiplier, **common_kwargs),
                          Conv2dNormActivation(multiplier, multiplier, **common_kwargs)) for _ in range(num_levels)])

        self.inputs_fusions, self.offsets_fusions = self.offsets_fusions[:1], self.offsets_fusions[1:]
        for _ in range(num_levels - 1):
            self.inputs_fusions.append(Conv2dNormActivation(multiplier * 2, multiplier, **common_kwargs))
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.warp_networks = nn.ModuleList([WarpNet(multiplier,
                                                    multiplier,
                                                    kernel_size=3,
                                                    padding=1,
                                                    deformable_groups=groups)
                                            for _ in range(num_levels)])
        self.output_fusion = nn.ModuleList([Conv2dNormActivation(multiplier * 2, multiplier, **common_kwargs),
                                            nn.Conv2d(multiplier * 2, multiplier, **conv2d_kwargs)])
        self.leaky_relu01 = LeakyReLU1EM1_()

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

        pyramid = [warp(level, offset) for warp, level, offset in zip(self.warp_networks, pyramid_a, offsets)]
        x = self.leaky_relu01(pyramid[0])
        for fusion, level in zip(self.output_fusion, pyramid[1:]):
            x = self.upsample2x(x)
            x = torch.cat((level, x), dim=1)
            x = fusion(x)
        return x
