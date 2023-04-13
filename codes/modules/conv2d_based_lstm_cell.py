import torch
import torch.nn as nn


class Conv2dBasedLSTMCell(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        super(Conv2dBasedLSTMCell, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels + out_channels,
                                out_channels=4 * out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias,
                                padding_mode=padding_mode,
                                device=device,
                                dtype=dtype)
        self.out_channels = out_channels
        self.device = device
        self.dtype = dtype

    def forward(self, x, state=None):
        h_curr, c_curr = torch.zeros(*(2, x.size(0), self.out_channels, *x.size()[2:]),
                                     requires_grad=True,
                                     device=self.device,
                                     dtype=self.dtype) if state is None else state

        x = torch.cat((x, h_curr), dim=1)
        x = self.conv2d(x)

        cc_i, cc_f, cc_o, cc_g = torch.split(x, self.out_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_curr + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
