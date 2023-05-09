from modules.zsm import ZSM
from .conv import adapt_conv


def adapt_decoder(weights):
    generator = ZSM(3, 3).decoder
    for i in range(40):
        generator[i].conv1.load_state_dict(adapt_conv(weights, f"recon_trunk.{i}.conv1"))
        generator[i].conv2.load_state_dict(adapt_conv(weights, f"recon_trunk.{i}.conv2"))

    generator[40][0].load_state_dict(adapt_conv(weights, "upconv1"))
    generator[41][0].load_state_dict(adapt_conv(weights, "upconv2"))
    generator[42][0].load_state_dict(adapt_conv(weights, "HRconv"))
    generator[43].load_state_dict(adapt_conv(weights, "conv_last"))
    return generator.state_dict()
