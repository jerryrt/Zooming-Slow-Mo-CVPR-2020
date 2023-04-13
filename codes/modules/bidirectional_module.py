import torch
import torch.nn as nn


class BidirectionalModule(nn.Module):
    def __init__(self, module):
        super(BidirectionalModule, self).__init__()
        self.module = module

    def forward(self, x):
        out = self.module(x)
        tuo = self.module(torch.flip(x, dims=[0]))
        return torch.cat((out, torch.flip(tuo, dims=[0])), dim=2)
