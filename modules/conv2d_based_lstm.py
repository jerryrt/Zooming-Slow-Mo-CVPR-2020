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
        self.out_channels = out_channels
        self.device = device
        self.dtype = dtype
        common_kwargs = dict(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=groups,
                             bias=bias,
                             padding_mode=padding_mode,
                             device=device,
                             dtype=dtype)
        self.cells = nn.ModuleList([Conv2dBasedLSTMCell(in_channels, out_channels, **common_kwargs)])
        for _ in range(self.num_layers - 1):
            self.cells.append(Conv2dBasedLSTMCell(out_channels, out_channels, **common_kwargs))

    def forward(self, x, hidden_state=None):
        out, history = [], []
        for i, module in enumerate(self.cells):
            state = torch.zeros(*(2, x[0].size(0), self.out_channels, *x[0].size()[2:]),
                                requires_grad=False,
                                device=x[0].device,
                                dtype=x[0].dtype) if hidden_state is None else hidden_state[i]
            cycle_out = []
            for t in range(len(x)):
                state = module(x[t], state)
                cycle_out.append(state[0])
            x = cycle_out
            out.append(x)
            history.append(state)

        return x, (out, history) if self.return_all_layers else (out[-1:], history[-1:])
