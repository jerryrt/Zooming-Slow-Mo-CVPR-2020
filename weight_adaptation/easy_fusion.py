from modules.zsm.easy_fusion import EasyFusion
from .conv import adapt_conv
from .pyramid_fusion import adapt_pyramid_fusion


def adapt_easy_fusion(weights, name_prefix):
    fusion = EasyFusion(64, 8)

    keys = [
        "fea_L2_conv1",
        "fea_L2_conv2",
        "fea_L3_conv1",
        "fea_L3_conv2"
    ]
    pruned_state_dict = {}
    for key in keys:
        pruned_state_dict[key] = adapt_conv(weights, f"{name_prefix}.{key}")

    fusion.pyramid_extractor.levels[0][0][0].load_state_dict(pruned_state_dict["fea_L2_conv1"])
    fusion.pyramid_extractor.levels[0][1][0].load_state_dict(pruned_state_dict["fea_L2_conv2"])

    fusion.pyramid_extractor.levels[1][0][0].load_state_dict(pruned_state_dict["fea_L3_conv1"])
    fusion.pyramid_extractor.levels[1][1][0].load_state_dict(pruned_state_dict["fea_L3_conv2"])

    fusion.pyramid_fusion.module_ab.load_state_dict(adapt_pyramid_fusion(weights, f"{name_prefix}.pcd_align"))
    fusion.pyramid_fusion.module_ba.load_state_dict(
        adapt_pyramid_fusion(weights, f"{name_prefix}.pcd_align", forward=False))
    fusion.pyramid_fusion.fusion.load_state_dict(adapt_conv(weights, f"{name_prefix}.fusion"))
    return fusion.state_dict()
