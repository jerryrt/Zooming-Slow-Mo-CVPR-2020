from .conv import adapt_conv


def adapt_output_fusion(weights):
    return adapt_conv(weights, "ConvBLSTM.conv_1x1")
