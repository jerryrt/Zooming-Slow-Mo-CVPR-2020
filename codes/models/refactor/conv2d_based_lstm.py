import torch
import torch.nn as nn
from .conv2d_based_lstm_cell import Conv2dBasedLSTMCell


class Conv2dBasedLSTM(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 return_all_layers=False,
                 device=None,
                 dtype=None):
        super(Conv2dBasedLSTM, self).__init__()

        self.return_all_layers = return_all_layers
        self.num_layers = num_layers
        common_kwargs = dict(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=groups,
                             bias=bias,
                             padding_mode=padding_mode,
                             device=device,
                             dtype=dtype)
        self.modules = nn.ModuleList([Conv2dBasedLSTMCell(in_channels, out_channels, **common_kwargs)])
        self.modules.extend([Conv2dBasedLSTMCell(out_channels, out_channels, **common_kwargs)
                             for _ in range(self.num_layers - 1)])

    def forward(self, x, hidden_state=None):
        out, history = [], []
        for i, module in enumerate(self.modules):
            state = None if hidden_state is None else hidden_state[i]
            cycle_out = []
            for t in range(len(x)):
                state = module(x[t], state)
                cycle_out.append(state[0])
            x = torch.stack(cycle_out, dim=0)
            out.append(x)
            history.append(state)

        if not self.return_all_layers:
            return out[-1:], history[-1:]
        return out, history
