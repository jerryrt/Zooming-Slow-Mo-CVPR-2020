from modules.zsm import ZSM
from .conv import adapt_conv



def adapt_feature_extractor(weights):
    fe = ZSM(3,3).feature_extractor#FeatureExtractor(3, 64, 5)

    print(fe[0][0].load_state_dict(adapt_conv(weights, "conv_first")))
    for i in range(5):
        fe[i+1].conv1.load_state_dict(adapt_conv(weights, f"feature_extraction.{i}.conv1"))
        fe[i+1].conv2.load_state_dict(adapt_conv(weights, f"feature_extraction.{i}.conv2"))
    return fe.state_dict()
