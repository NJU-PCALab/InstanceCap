import av
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import numpy as np
import json
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates
from PIL import Image
import copy
import sys
import warnings
from decord import VideoReader, cpu
import os
import glob
import logging
from tqdm import tqdm

device = "cuda"

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--llava-next-video-path", type=str, default="/root/data/weights/LLaVA-Video-72B-Qwen2/")
parser.add_argument("--video-base", type=str, default="")
parser.add_argument("--output", type=str, default="./results/ours/single")
parser.add_argument("--dimensions", type=str, nargs="+", default=["action", "color", "shape", "texture", "detail"])
args = parser.parse_args()

# Load llava-next-video
pretrained = args.llava_next_video_path
model_name = "llava_qwen"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name,
    torch_dtype="bfloat16", device_map=device_map, ignore_mismatched_sizes=True
)
conv_template = "qwen_1_5"
model.eval().type(torch.bfloat16)

device = "cuda"

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    sampling_interval = round(vr.get_avg_fps() / fps)
    frame_idx = list(range(0, len(vr), sampling_interval))
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_time_str, video_time

def extract_substring_after_last_occurrence(output, s):
    last_index = output.rfind(s)
    if last_index == -1:
        return output
    start_index = last_index + len(s)
    return output[start_index:].strip()

def build_video_prompt(image_processor, video, frame_time, video_time):
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device).bfloat16()
    video = [video]
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. "
        f"These frames are located at {frame_time}. Please answer the following questions related to this video."
    )
    pre_instruction = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n"
    return video, pre_instruction

def get_answer(model, tokenizer, video, conv):
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    answer = extract_substring_after_last_occurrence(text_output, "assistant\n")
    return answer

def is_answer_yes_no(answer):
    normalized_answer = answer.strip().upper()
    if "YES" in normalized_answer:
        return True
    elif "NO" in normalized_answer:
        return False
    else:
        return False

def handle_dimension(dimension_data, video_folder_path, dimension_type):
    logger = logging.getLogger(dimension_type)
    logger.setLevel(logging.INFO)

    # Create handlers
    results_folder = args.output
    dimension_result_path = os.path.join(results_folder, f"{dimension_type}_results.txt")
    file_handler = logging.FileHandler(filename=dimension_result_path)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create formatters and add to handlers
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.info(f"Processing {dimension_type} dimension...")

    score = 0
    total_entries = len(dimension_data)

    # Ensure the video files match the data entries
    video_files = sorted(glob.glob(os.path.join(video_folder_path, "*.mp4")))
    if not video_files:
        logger.error(f"No video files found in {video_folder_path}.")
        return 0.0

    if len(video_files) != total_entries:
        logger.warning(f"Number of video files ({len(video_files)}) does not match data entries ({total_entries}).")

    for i, entry in tqdm(enumerate(dimension_data), total=total_entries):
        sentence = entry.get("sentence", "")
        instance = entry.get("instance", {})
        instance_class = instance.get("class", "")
        instance_specific = instance.get(dimension_type, "")

        if i < len(video_files):
            video_file = video_files[i]
        else:
            logger.error(f"No corresponding video file for entry {i}.")
            continue

        # Load video
        try:
            original_video, original_frame_time, original_video_time = load_video(video_file, 8, 1, force_sample=True)
        except Exception as e:
            logger.error(f"Failed to load video {video_file}: {e}")
            continue

        original_video, original_pre_instruction = build_video_prompt(image_processor, original_video, original_frame_time, original_video_time)

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], original_pre_instruction)
        conv.append_message(conv.roles[0], f"""Describe this video in one sentence, care about the {dimension_type}.""")

        answer = get_answer(model, tokenizer, original_video, conv)
        logger.info(f"Video: {video_file}")
        logger.info(f"Sentence: {sentence}")
        logger.info(f"Generated Caption: {answer}")

        conv.append_message(conv.roles[1], answer)

        tmp_conv = conv.copy()
        tmp_conv.append_message(conv.roles[0], (
            f"Tell me if '{instance_class}' is in the video?"
            f"Note the synonyms of '{instance_class}'. Your answer can only be YES or NO. "
            f"Do not output any answer that is not YES or NO."
        ))
        presence_answer = get_answer(model, tokenizer, original_video, tmp_conv)
        logger.info(f"Presence of '{instance_class}' in the video: {presence_answer}")

        if not is_answer_yes_no(presence_answer):
            logger.info(f"Unable to determine presence of '{instance_class}' in the video.")
            continue

        conv.append_message(conv.roles[0], (
            f"Based on your previous answer, tell me what is the '{dimension_type}' of '{instance_class}' in the video?"
            "Be careful to ignore camera movement."
        ))
        dimension_answer = get_answer(model, tokenizer, original_video, conv)
        conv.append_message(conv.roles[1], dimension_answer)
        logger.info(f"Expected {dimension_type.capitalize()}: {instance_specific}")
        logger.info(f"Model's {dimension_type.capitalize()} Answer: {dimension_answer}")

        if dimension_type == "detail":
            conv.append_message(conv.roles[0], (
                f"Is the '{instance_specific}' of '{instance_class}' completely reflected in the video?"
                "Your answer can only be YES or NO. Do not output any answer that is not YES or NO."
            ))
        else:
            conv.append_message(conv.roles[0], (
                f"Do you think the '{dimension_type}' of '{instance_class}' in the video completely matches to '{instance_specific}'? "
                "Your answer can only be YES or NO. Do not output any answer that is not YES or NO."
            ))

        final_answer = get_answer(model, tokenizer, original_video, conv)
        conv.append_message(conv.roles[1], final_answer)
        logger.info(f"Final Answer: {final_answer}")

        is_correct = is_answer_yes_no(final_answer)
        score += int(is_correct)
        logger.info(f"Score for this entry: {int(is_correct)}")
        logger.info("=" * 60)

    score_ratio = score / total_entries if total_entries > 0 else 0.0
    logger.info(f"{dimension_type.capitalize()} Dimension Score: {score_ratio:.4f}")

    # Remove handlers after processing
    logger.removeHandler(file_handler)
    logger.removeHandler(stream_handler)

    return score_ratio

def main(args):
    dimensions = args.dimensions
    all_scores = {}

    # Map dimensions to their respective handlers
    dimension_functions = {
        "action": "action",
        "color": "color",
        "detail": "detail",
        "shape": "shape",
        "texture": "texture",
    }

    results_folder = args.output
    import shutil
    # If the folder exists, remove it
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)

    # Create a new results folder
    os.makedirs(results_folder, exist_ok=True)

    for dimension in dimensions:
        if dimension in dimension_functions:
            # Load dimension data from the respective JSONL files
            dimension_file_path = os.path.join("dimensions", "single", f"{dimension}.jsonl")
            video_folder_path = os.path.join(args.video_base, "single", dimension)

            try:
                with open(dimension_file_path, 'r', encoding='utf-8') as f:
                    dimension_data = [json.loads(line) for line in f]
            except FileNotFoundError:
                print(f"Error: The file {dimension_file_path} does not exist.")
                continue
            except Exception as e:
                print(f"An error occurred while reading {dimension_file_path}: {e}")
                continue

            # Process the dimension data and get the score
            score = handle_dimension(dimension_data, video_folder_path, dimension)
            all_scores[dimension] = score

    if all_scores:
        # Write all dimension scores to all_results.txt
        all_results_path = os.path.join(results_folder, "all_results.txt")
        with open(all_results_path, "w") as f:
            for dimension, score in all_scores.items():
                f.write(f"{dimension.capitalize()} Dimension Score: {score:.4f}\n")
                print(f"{dimension.capitalize()} Dimension Score: {score:.4f}")

if __name__ == "__main__":
    main(args)
