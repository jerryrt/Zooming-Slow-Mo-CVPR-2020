import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class WarpNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 deformable_groups,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(WarpNet, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.deformable_groups = deformable_groups
        self.hidden_dim = self.deformable_groups * kernel_size[0] * kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(in_channels,
                                          3 * self.hidden_dim,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          bias=True)
        self.deform_conv2d = DeformConv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)

    def forward(self, input, offsets):
        x = self.conv_offset_mask(offsets)
        offset, mask = torch.split(x, 2 * self.hidden_dim, dim=1)
        mask = torch.sigmoid(mask)
        return self.deform_conv2d(input, offset, mask)
