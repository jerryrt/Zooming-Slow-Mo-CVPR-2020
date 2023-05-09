from modules.zsm import ZSM

from .output_fusion import adapt_output_fusion
from .feature_extractor import adapt_feature_extractor
from .easy_fusion import adapt_easy_fusion
from .main_block import adapt_main_block
from .decoder import adapt_decoder


def adapt_weights(weights):
    a = ZSM(3, 3)
    a.feature_extractor.load_state_dict(adapt_feature_extractor(weights))
    a.fusion.load_state_dict(adapt_easy_fusion(weights, ""))
    a.main_module.module[0].load_state_dict(adapt_main_block(weights))
    a.output_fusion.load_state_dict(adapt_output_fusion(weights))
    a.decoder.load_state_dict(adapt_decoder(weights))
    return a.state_dict()


# weights = torch.load("LunaTokis.pth")
# torch.save(a.state_dict(), "../checkpoint/zsm.pt")
