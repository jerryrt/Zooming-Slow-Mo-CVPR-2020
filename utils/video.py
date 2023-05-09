import cv2
import math
import numpy as np
import os


def get_video_size(path):
    capture = cv2.VideoCapture(path)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    return size


def pad_video_to_be_multiple_of_factor(input_path, output_path, factor=4):
    r = factor
    size = np.array(get_video_size(input_path))
    size = (r * np.ceil(size / r)).astype(int)
    w, h = size
    i, o = input_path, output_path
    x, y = (0, 0)
    os.system(f"""ffmpeg -i '{i}' -vf "pad=width={w}:height={h}:x={x}:y={y}:color=black" '{o}'""")


def crop_video(input_path, output_path, rectangle):
    l, t = rectangle[:2]
    w, h = np.array(rectangle[2:]) - np.array(rectangle[:2])
    os.system(f"""ffmpeg - i ${input_path} - filter: v "crop=${w}:${h}:${l}:${t}" ${output_path}""")

