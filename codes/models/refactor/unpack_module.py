import torch.nn as nn


class UnpackModule(nn.Module):
    def __init__(self, module, idx=()):
        super(UnpackModule, self).__init__()
        self.module = module
        self.idx = idx

    def forward(self, x):
        return self.module(x)[self.idx]
