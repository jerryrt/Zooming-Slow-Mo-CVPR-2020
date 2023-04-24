from collections import OrderedDict


def adapt_conv(weights, prefix):
    state_dict = OrderedDict()
    pruned_key = prefix.replace(".weight", "").replace(".bias", "")
    if pruned_key[0] == ".":
        pruned_key = pruned_key[1:]
    state_dict["weight"] = weights[f"{pruned_key}.weight"]
    state_dict["bias"] = weights[f"{pruned_key}.bias"]
    return state_dict
