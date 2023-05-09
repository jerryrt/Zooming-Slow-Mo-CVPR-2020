from weight_adaptation import adapt_zsm_weights
import torch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", type=str, required=True, default="LunaTokis.pth",
                        help="path to weights from old version.")
    parser.add_argument('-o', "--output", type=str, default="zsm.pth", help="output path.")
    args = parser.parse_args()
    torch.save(adapt_zsm_weights(torch.load(args.input)), args.output)
