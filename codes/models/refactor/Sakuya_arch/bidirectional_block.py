import torch
import torch.nn as nn
from .basic_block import BasicBlock


class BidirectionalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers,
                 fusion_groups,
                 batch_first=False,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 return_all_layers=False,
                 device=None,
                 dtype=None):
        super(BidirectionalBlock, self).__init__()
        self.module = BasicBlock(**locals())
        self.fusion = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        out = self.module(x)[0]
        tuo = self.module(torch.flip(x, dims=[0]))[0]
        x = torch.cat((out, torch.flip(tuo, dims=[0])), dim=2)
        return torch.stack([self.fusion(x[t]) for t in range(len(x))], dim=0)
