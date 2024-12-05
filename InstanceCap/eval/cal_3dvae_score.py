import argparse
import logging
import time
import imageio
import torch
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def sort_video_files(file_list):
    return sorted(file_list, key=natural_sort_key)

def get_and_sort_video_files(directory):
    video_files = [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.mp4')]
    return sort_video_files(video_files)

def encode_video(model, video_path):
    video_reader = imageio.get_reader(video_path, "ffmpeg")

    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将 numpy 数组转换为 PIL 图像
        transforms.Resize((224, 224)),  # 调整图像大小到 224x224
        transforms.ToTensor(),  # 转换为 PyTorch 张量
    ])

    frames = [transform(frame) for frame in video_reader]
    frame_indices = np.linspace(0, len(frames) - 1, num=8).astype(int)
    frames = [frames[i] for i in frame_indices]
    video_reader.close()

    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)

    with torch.no_grad():
        encoded_frames = model.encode(frames_tensor)[0].sample()
    return encoded_frames

def cal_3dvae_score(model, ori_video_path, gen_video_path):
    encoded_ori = encode_video(model, ori_video_path)
    encoded_gen = encode_video(model, gen_video_path)

    return torch.norm(encoded_ori - encoded_gen, p=2).cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='./cogvideox/CogVideoX-5b/vae')
    parser.add_argument("--ori_videos", type=str, default='',
                        help="path to ori_videos")
    parser.add_argument("--gen_videos", type=str, default='',
                        help="path to gen_videos")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    model = AutoencoderKLCogVideoX.from_pretrained(args.model_path, torch_dtype=dtype).to(device)

    model.enable_slicing()
    model.enable_tiling()

    ori_videos = args.ori_videos
    gen_videos = args.gen_videos

    ori_video_paths = get_and_sort_video_files(ori_videos)
    gen_video_paths = get_and_sort_video_files(gen_videos)

    # set up logging +++++++++++++++++++++++++++++++++
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f".results", exist_ok=True)
    # Set up logging
    log_file_path = f"./results/3dvae_record.txt"
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"./results/3dvae_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)
    # set up logging +++++++++++++++++++++++++++++++++
    # Load pretrained models

    scores = []
    average_score = 0
    for i in tqdm(range(len(ori_video_paths))):
        ori_video_path = ori_video_paths[i]
        gen_video_path = gen_video_paths[i]

        score = cal_3dvae_score(model, ori_video_path, gen_video_path)
        scores.append(score)
        average_score = sum(scores) / len(scores)

        logging.info(
            f"Vid:{os.path.basename(ori_video_path)} and {os.path.basename(gen_video_path)},  Current 3D VAE score: {score}, Current avg. 3D VAE score: {average_score},  ")

    logging.info(f"Final 3D VAE score: {average_score}, Total videos: {len(scores)}")