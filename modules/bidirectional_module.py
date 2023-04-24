import torch
import torch.nn as nn


class BidirectionalModule(nn.Module):
    def __init__(self, module):
        super(BidirectionalModule, self).__init__()
        self.module = module

    def forward(self, x):
        out = self.module(x)
        tuo = self.module(x[::-1]) if isinstance(x, list) else self.module(torch.flip(x, dims=[0]))
        n = len(out)
        return [torch.cat((out[i], tuo[n-i-1]), dim=1) for i in range(n)]
