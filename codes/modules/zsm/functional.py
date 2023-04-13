import torch.nn as nn
import functools

LeakyReLU1EM1 = functools.partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)
