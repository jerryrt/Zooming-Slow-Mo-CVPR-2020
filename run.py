import numpy as np
import torch
import torch.nn as nn
import time
import os
import math
import cv2

#  ffmpeg -i input.avi -vf "pad=width=640:height=480:x=0:y=0:color=black" output.avi


def preprocess(x: np.ndarray):
    x = x[..., ::-1]  # (reorder color channels) BGR -> RGB
    x = x / 255  # (range change) [0, 255] -> [0, 1]
    x = x.transpose(0, 3, 1, 2)
    h_n = int(4 * np.ceil(x.shape[2] / 4))
    w_n = int(4 * np.ceil(x.shape[3] / 4))
    shape = x.shape
    shape = shape[:-2] + (h_n, w_n)
    # shape[-1] = w_n
    # shape[-2] = h_n
    imgs_temp = np.zeros(shape)
    imgs_temp[:, :, 0:x.shape[-2], 0:x.shape[-1]] = x
    return imgs_temp  # (reshape) B x H x W x C -> B x C x H x W


def postprocessing(x: np.ndarray):
    x = x.transpose(0, 2, 3, 1)  # (reshape) B x C x H x W -> B x H x W x C
    # x = np.array(np.clip(x * 255, 0, 255) + 0.5, dtype=np.uint8)  # (range change with rounding) [0, 1] -> [0, 255]

    x = np.array(np.clip(x * 255, 0, 255) + 0.5, dtype=np.uint8)  # (range change with rounding) [0, 1] -> [0, 255]
    return x[..., ::-1]


@torch.no_grad()
def run_on_video(input_path, output_path, model, batch_size=1, device="cuda") -> None:
    point_time = time.time()
    assert os.path.isfile(input_path), f"{input_path} not found."

    reader = cv2.VideoCapture(input_path)
    fps = int(math.ceil(reader.get(cv2.CAP_PROP_FPS)))
    size = (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    r = model.upscale_factor
    size = (r * int(r * np.ceil(size[0] / r)), r * int(r * np.ceil(size[1] / r)))

    writer = cv2.VideoWriter(filename=output_path,
                             fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                             fps=fps * 2,
                             frameSize=size,
                             isColor=True)

    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    processing_stats = [0]

    frame_no = 0
    status, frame = reader.read()
    if not status:
        return
    # writer.write(frame)
    frames = [frame]
    i = 0
    while status:# and i<220:
        i+=1
        batch_time = time.time()
        for _ in range(batch_size):
            frame_no += 1
            status, frame = reader.read()
            if not status:
                break
            frames.append(frame)
        frames = frames[-batch_size - 1:]
        x = preprocess(np.stack(frames))[:, None]
        x = torch.Tensor(x).to(device)
        predicted_patch = model(x)#.cpu().detach().numpy()
        for frame in predicted_patch[1:]:
            x = frame.cpu().detach().numpy()
            x = postprocessing(x)[0]
            writer.write(x)

        step_time = time.time() - batch_time
        processing_stats.append(step_time)
        processing_stats = processing_stats[-10:]
        print(f"total time: "
              f"{time.time() - point_time:0.2f}/~{frame_count / batch_size * np.mean(processing_stats):0.2f} sec., "
              f"total frames: {str(frame_no).zfill(3)}/{str(frame_count).zfill(3)} frames.")
    reader.release(), writer.release()
    print(f"pass {time.time() - point_time:0.2f} sec.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", type=str, required=True, default="input.mp4", help="input video path.")
    parser.add_argument('-o', "--output", type=str, default="output.mp4", help="output video path.")
    parser.add_argument("--device", default='cuda', choices=["cuda", "cpu"], help="device.")
    parser.add_argument("--batch", default=1, type=int, help="device.")
    args = parser.parse_args()

    from modules import ZSM

    a = ZSM(3, 3)
    a.load_state_dict(torch.load("checkpoint/zsm.pt"))
    a.to(args.device)
    a.eval()

    run_on_video(args.input, args.output, a, batch_size=args.batch, device=args.device)
