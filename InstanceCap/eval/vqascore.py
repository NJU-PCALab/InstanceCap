import argparse
import logging
import os
import time

import cv2
import torch
import re
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def sort_video_files(file_list):
    return sorted(file_list, key=natural_sort_key)

def get_and_sort_video_files(directory):
    video_files = [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.mp4')]
    return sort_video_files(video_files)

import os
from PIL import Image

def get_score(model, images, text):
    # 确保temp文件夹存在
    temp_dir = f'temp{args.num}'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 保存每个图像到temp文件夹，并记录文件路径
    image_paths = []
    for i, image in enumerate(images):
        # 将 NumPy 数组转换为 PIL 图像
        pil_image = Image.fromarray(image)
        image_path = os.path.join(temp_dir, f'image_{i}.png')
        pil_image.save(image_path, 'PNG')
        image_paths.append(image_path)

    # 将文件路径列表传递给model
    score = model(images=image_paths, texts=[text])

    # 清理临时文件（可选）
    for image_path in image_paths:
        os.remove(image_path)

    return score.cpu().numpy().mean()

def calculate_vqa_score(video_path, text, model):
    # 加载视频
    cap = cv2.VideoCapture(video_path)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算采样间隔
    num_samples = 6
    interval = max(total_frames // num_samples, 1)  # 确保间隔至少为1

    # 从视频中均匀采样帧
    frames = []
    for i in range(num_samples):
        # 设置帧位置
        frame_id = min(i * interval, total_frames - 1)  # 避免超出范围
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧从 BGR 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    # 释放视频捕获对象
    cap.release()

    # 调用 get_score 函数计算分数
    return get_score(model, frames, text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='',
                        help="path to videos")
    parser.add_argument("--prompts", type=str,
                        default='',
                        help="path to prompts")
    parser.add_argument("--num", type=int,
                        default=0,
                        help="path to prompts")
    args = parser.parse_args()

    dir_videos = args.dir_videos
    prompts_txt = args.prompts

    video_paths = get_and_sort_video_files(dir_videos)
    with open(prompts_txt, 'r') as file:
        prompts = [line.rstrip('\n') for line in file]

    # set up logging +++++++++++++++++++++++++++++++++
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"results", exist_ok=True)
    # Set up logging
    log_file_path = f"./results/vqa_record.txt"
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"./results/vqa_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)
    # set up logging +++++++++++++++++++++++++++++++++
    # Load pretrained models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import t2v_metrics
    model =  t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model
    
    scores = []
    average_score = 0
    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        prompt = prompts[i]

        score = calculate_vqa_score(video_path, prompt, model)
        scores.append(score)
        average_score = sum(scores) / len(scores)

        logging.info(
            f"Vid: {os.path.basename(video_path)},  Current VQA score: {score}, Current avg. VQA score: {average_score},  current results: {args.prompts.split('/')[-1].split('.')[0]}")

    logging.info(f"Final VQA score: {average_score}, Total videos: {len(scores)}")

