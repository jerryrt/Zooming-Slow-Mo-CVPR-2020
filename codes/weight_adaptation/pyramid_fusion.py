from .conv import adapt_conv
from codes.modules.zsm.pyramid_fusion import PyramidFusion


def adapt_pyramid_fusion(weights, name_prefix, forward=True):
    kind = 1 if forward else 2
    pf = PyramidFusion(64, 8)

    keys = [
        "L3_offset_conv1_1",
        "L3_offset_conv2_1",
        "L3_dcnpack_1",
        "L2_offset_conv1_1",
        "L2_offset_conv2_1",
        "L2_offset_conv3_1",
        "L2_dcnpack_1",
        "L2_fea_conv_1",
        "L1_offset_conv1_1",
        "L1_offset_conv2_1",
        "L1_offset_conv3_1",
        "L1_dcnpack_1",
        "L1_fea_conv_1",
        "L3_offset_conv1_2",
        "L3_offset_conv2_2",
        "L3_dcnpack_2",
        "L2_offset_conv1_2",
        "L2_offset_conv2_2",
        "L2_offset_conv3_2",
        "L2_dcnpack_2",
        "L2_fea_conv_2",
        "L1_offset_conv1_2",
        "L1_offset_conv2_2",
        "L1_offset_conv3_2",
        "L1_dcnpack_2",
        "L1_fea_conv_2"
    ]
    pruned_state_dict = {}
    for key in keys:
        pruned_state_dict[key] = adapt_conv(weights, f"{name_prefix}.{key}")

    for i in range(3):
        code = 3 - i

        # deform_conv2d_dict = pf.warp_networks[i].deform_conv2d.state_dict()
        # deform_conv2d_dict['weight'] = pruned_state_dict[f"L{code}_dcnpack_{kind}"]["weight"]
        # deform_conv2d_dict['bias'] = pruned_state_dict[f"L{code}_dcnpack_{kind}"]["bias"]
        # pf.warp_networks[i].deform_conv2d.load_state_dict(deform_conv2d_dict)
        #
        pf.warp_networks[i].deform_conv2d.load_state_dict(pruned_state_dict[f"L{code}_dcnpack_{kind}"])
        #

        #
        offsets_state_dict = adapt_conv(weights, f"{name_prefix}.L{code}_dcnpack_{kind}.conv_offset_mask")
        #
        pf.warp_networks[i].conv_offset_mask.load_state_dict(offsets_state_dict)

        # conv_offset_mask_state_dict = pf.warp_networks[i].conv_offset_mask.state_dict()
        # conv_offset_mask_state_dict['weight'] = pruned_state_dict[f"L{code}_dcnpack_{kind}"]["conv_offset_mask.weight"]
        # conv_offset_mask_state_dict['bias'] = pruned_state_dict[f"L{code}_dcnpack_{kind}"]["conv_offset_mask.bias"]
        # pf.warp_networks[i].conv_offset_mask.load_state_dict(conv_offset_mask_state_dict)

    pf.output_fusion[0][0].load_state_dict(pruned_state_dict[f"L2_fea_conv_{kind}"])
    pf.output_fusion[1][0].load_state_dict(pruned_state_dict[f"L1_fea_conv_{kind}"])

    pf.inputs_fusions[0][0][0].load_state_dict(pruned_state_dict[f"L3_offset_conv1_{kind}"])
    pf.inputs_fusions[0][1][0].load_state_dict(pruned_state_dict[f"L3_offset_conv2_{kind}"])
    pf.inputs_fusions[1][0].load_state_dict(pruned_state_dict[f"L2_offset_conv1_{kind}"])
    pf.inputs_fusions[2][0].load_state_dict(pruned_state_dict[f"L1_offset_conv1_{kind}"])

    pf.offsets_fusions[0][0][0].load_state_dict(pruned_state_dict[f"L2_offset_conv2_{kind}"])
    pf.offsets_fusions[0][1][0].load_state_dict(pruned_state_dict[f"L2_offset_conv3_{kind}"])

    pf.offsets_fusions[1][0][0].load_state_dict(pruned_state_dict[f"L1_offset_conv2_{kind}"])
    pf.offsets_fusions[1][1][0].load_state_dict(pruned_state_dict[f"L1_offset_conv3_{kind}"])
    return pf.state_dict()
