from ..conv2d_based_lstm import Conv2dBasedLSTM
from .easy_fusion import EasyFusion


class MainBlock(Conv2dBasedLSTM):
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
        Conv2dBasedLSTM.__init__(self,  # **locals())
                                 in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 num_layers=num_layers,
                                 # batch_first=batch_first,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias,
                                 padding_mode=padding_mode,
                                 return_all_layers=return_all_layers,
                                 device=device,
                                 dtype=dtype)

        self.fusion_h = EasyFusion(multiplier=in_channels, groups=fusion_groups)
        self.fusion_c = EasyFusion(multiplier=in_channels, groups=fusion_groups)

    def forward(self, x, hidden_state=None):
        out, history = [], []
        for i, module in enumerate(self.modules):
            state = None if hidden_state is None else hidden_state[i]
            cycle_out = []
            for t in range(len(x)):
                state = self.fusion_h(x[t], state[0]), self.fusion_c(x[t], state[1])
                state = module(x[t], state)
                cycle_out.append(state[0])
            # x = torch.stack(cycle_out, dim=0)
            x = cycle_out
            out.append(x)
            history.append(state)

        return x, (out, history) if self.return_all_layers else (out[-1:], history[-1:])
