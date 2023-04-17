from codes.modules.zsm.main_block import MainBlock
from .conv import adapt_conv
from .easy_fusion import adapt_easy_fusion


def adapt_main_block(weight):
    mb = MainBlock(in_channels=64,
                   out_channels=64,
                   kernel_size=3,
                   padding=1,
                   num_layers=1,
                   fusion_groups=8,
                   return_all_layers=False)

    mb.cells[0].conv2d.load_state_dict(adapt_conv(weight, 'ConvBLSTM.forward_net.cell_list.0.conv'))
    mb.fusion_h.load_state_dict(adapt_easy_fusion(weight, 'ConvBLSTM.forward_net.pcd_h'))
    mb.fusion_c.load_state_dict(adapt_easy_fusion(weight, 'ConvBLSTM.forward_net.pcd_c'))
    return mb.state_dict()
