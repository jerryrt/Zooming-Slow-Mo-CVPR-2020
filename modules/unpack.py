import torch.nn as nn


class Unpack(nn.Module):
    def __init__(self, indexes=()):
        super(Unpack, self).__init__()
        self.indexes = [indexes] if isinstance(indexes, int) else indexes

    def forward(self, x):
        for idx in self.indexes:
            x = x[idx]
        return x
