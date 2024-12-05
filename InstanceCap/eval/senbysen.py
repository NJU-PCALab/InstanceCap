import argparse
import logging
import os
import time

import cv2
import torch
from sentencex import segment
import re
from tqdm import tqdm
from transformers import CLIPModel, AutoTokenizer

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def sort_video_files(file_list):
    return sorted(file_list, key=natural_sort_key)

def get_and_sort_video_files(directory):
    video_files = [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.mp4')]
    return sort_video_files(video_files)

def calculate_clip_score(video_path, text, model, tokenizer):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Extract frames from the video
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame, (224, 224))  # Resize the frame to match the expected input size
        frames.append(resized_frame)

    # Convert numpy arrays to tensors, change dtype to float, and resize frames
    tensor_frames = [torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames]

    # Initialize an empty tensor to store the concatenated features
    concatenated_features = torch.tensor([], device=device)

    # Generate embeddings for each frame and concatenate the features
    with torch.no_grad():
        for frame in tensor_frames:
            frame_input = frame.unsqueeze(0).to(device)  # Add batch dimension and move the frame to the device
            frame_features = model.get_image_features(frame_input)
            concatenated_features = torch.cat((concatenated_features, frame_features), dim=0)

    # Tokenize the text
    text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)

    # Convert the tokenized text to a tensor and move it to the device
    text_input = text_tokens["input_ids"].to(device)

    # Generate text embeddings
    with torch.no_grad():
        text_features = model.get_text_features(text_input)

    # Calculate the cosine similarity scores
    concatenated_features = concatenated_features / concatenated_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    clip_score_frames = concatenated_features @ text_features.T
    # Calculate the average CLIP score across all frames, reflects temporal consistency
    clip_score_frames_avg = clip_score_frames.mean().item()

    return clip_score_frames_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, default='',
                        help="path to videos")
    parser.add_argument("--prompts", type=str,
                        default='',
                        help="path to prompts")
    args = parser.parse_args()

    dir_videos = args.dir_videos
    prompts_txt = args.prompts

    video_paths = get_and_sort_video_files(dir_videos)
    with open(prompts_txt, 'r') as file:
        prompts = [line.rstrip('\n') for line in file]

    # set up logging +++++++++++++++++++++++++++++++++
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f".results", exist_ok=True)
    # Set up logging
    log_file_path = f"./results/meaningless_record.txt"
    # Delete the log file if it exists
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(filename=f"./results/meaningless_record.txt")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    # Stream handler for displaying logs in the terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(stream_handler)
    # set up logging +++++++++++++++++++++++++++++++++
    # Load pretrained models
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = CLIPModel.from_pretrained("../checkpoints/clip-vit-base-patch32").to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained("../checkpoints/clip-vit-base-patch32")

    scores = []
    average_score = 0
    for i in tqdm(range(len(video_paths))):
        video_path = video_paths[i]
        prompt = prompts[i]

        seg_prompt = list(segment("en", prompt))
        score = 0
        for sentence in seg_prompt:
            score += calculate_clip_score(video_path, sentence, clip_model, clip_tokenizer)
        score /= len(seg_prompt)
        scores.append(score)
        average_score = sum(scores) / len(scores)

        logging.info(
            f"Vid: {os.path.basename(video_path)},  Current meaningful score: {score}, Current avg. meaningful score: {average_score},  ")

    logging.info(f"Final meaningful score: {average_score}, Total videos: {len(scores)}")