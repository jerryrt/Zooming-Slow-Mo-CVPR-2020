import numpy as np
import torch
import torch.nn as nn
import time
import os
import math
import cv2
import tempfile
from utils.video import pad_video_to_be_multiple_of_factor, crop_video
from tqdm import tqdm


def preprocess(x: np.ndarray):
    x = x[..., ::-1]  # (reorder color channels) BGR -> RGB
    x = x / 255  # (range change) [0, 255] -> [0, 1]
    x = x.transpose(0, 3, 1, 2)
    h_n = int(4 * np.ceil(x.shape[2] / 4))
    w_n = int(4 * np.ceil(x.shape[3] / 4))
    shape = x.shape
    shape = shape[:-2] + (h_n, w_n)
    imgs_temp = np.zeros(shape)
    imgs_temp[:, :, 0:x.shape[-2], 0:x.shape[-1]] = x
    return imgs_temp  # (reshape) B x H x W x C -> B x C x H x W


def fast_preprocess(x: np.ndarray):
    x = x[..., ::-1]  # (reorder color channels) BGR -> RGB
    x = x / 255  # (range change) [0, 255] -> [0, 1]
    x = x.transpose(0, 3, 1, 2)
    return x


def postprocessing(x: np.ndarray):
    x = x.transpose(0, 2, 3, 1)  # (reshape) B x C x H x W -> B x H x W x C
    # x = np.array(np.clip(x * 255, 0, 255) + 0.5, dtype=np.uint8)  # (range change with rounding) [0, 1] -> [0, 255]

    x = np.array(np.clip(x * 255, 0, 255) + 0.5, dtype=np.uint8)  # (range change with rounding) [0, 1] -> [0, 255]
    return x[..., ::-1]


@torch.no_grad()
def run_on_video(input_path, output_path, model, batch_size=1, device="cuda") -> None:
    assert os.path.isfile(input_path), f"{input_path} not found."

    model.to(args.device)
    model.eval()
    with tempfile.TemporaryDirectory() as directory:
        temp_input_path, temp_output_path = os.path.join(directory, "input.mp4"), os.path.join(directory, "output.mp4")

        pad_video_to_be_multiple_of_factor(input_path, temp_input_path, factor=4)
        reader = cv2.VideoCapture(temp_input_path)
        fps = int(math.ceil(reader.get(cv2.CAP_PROP_FPS)))
        w, h = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        r = model.upscale_factor
        size = (r * int(r * np.ceil(w / r)), r * int(r * np.ceil(h / r)))

        writer = cv2.VideoWriter(filename=temp_output_path,
                                 fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps=fps * 2,
                                 frameSize=size,
                                 isColor=True)

        frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        processing_bar = tqdm(total=frame_count)

        frame_no = 0
        status, frame = reader.read()
        if not status:
            return
        # writer.write(frame)
        frames = [frame]
        while status:
            for _ in range(batch_size):
                frame_no += 1
                status, frame = reader.read()
                if not status:
                    break
                frames.append(frame)
            frames = frames[-batch_size - 1:]
            x = fast_preprocess(np.stack(frames))[:, None]
            x = torch.Tensor(x).to(device)
            predicted_patch = model(x)
            for frame in predicted_patch[1:]:
                x = frame.cpu().detach().numpy()
                x = postprocessing(x)[0]
                writer.write(x)
            processing_bar.update(batch_size)
        crop_video(temp_output_path, output_path, [0, 0, w, h])
        reader.release(), writer.release(), processing_bar.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", type=str, required=True, default="input.mp4", help="input video path.")
    parser.add_argument('-o', "--output", type=str, default="output.mp4", help="output video path.")
    parser.add_argument("--device", default='cuda', choices=["cuda", "cpu"], help="device.")
    parser.add_argument("--batch", default=1, type=int, help="device.")
    args = parser.parse_args()

    from modules import zsm, ZSM_Weights


    class PlaceHolder:

        def get_state_dict(self, *args, **kwargs):
            return torch.load("checkpoint/zsm.pt")


    a = zsm(weights=PlaceHolder())

    run_on_video(args.input, args.output, a, batch_size=args.batch, device=args.device)
