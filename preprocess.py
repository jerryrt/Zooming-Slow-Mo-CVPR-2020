import cv2
import math
import numpy as np
import os
from utils.video import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", type=str, required=True)
    parser.add_argument('-o', "--output", type=str, default="outpu.mp4", help="output path.")
    args = parser.parse_args()

    pad_video_to_be_multiple_of_factor(args.input, args.output, 4)

