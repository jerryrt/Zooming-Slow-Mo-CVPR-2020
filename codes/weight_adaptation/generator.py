from .conv import adapt_conv
from codes.modules.zsm.resnet_based_generator import ResnetBasedGenerator as Generator


def adapt_generator(weights):
    generator = Generator(64, 3, 40)
    for i in range(40):
        generator[i].conv1.load_state_dict(adapt_conv(weights, f"recon_trunk.{i}.conv1"))
        generator[i].conv2.load_state_dict(adapt_conv(weights, f"recon_trunk.{i}.conv2"))

    generator[40].load_state_dict(adapt_conv(weights, "upconv1"))
    generator[43].load_state_dict(adapt_conv(weights, "upconv2"))
    generator[46].load_state_dict(adapt_conv(weights, "HRconv"))
    generator[48].load_state_dict(adapt_conv(weights, "conv_last"))
    return generator.state_dict()
