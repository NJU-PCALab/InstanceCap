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
parser.add_argument("--output", type=str, default="./results/ours/multiple")
parser.add_argument("--dimensions", type=str, nargs="+", default=["action", "color", "shape", "texture", "detail"])
args = parser.parse_args()

# Load LLaVA-Next-Video model
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
    sampling_interval = max(1, round(vr.get_avg_fps() / fps))
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
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    results_folder = args.output
    dimension_result_path = os.path.join(results_folder, f"{dimension_type}_results.txt")
    file_handler = logging.FileHandler(filename=dimension_result_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Processing {dimension_type} dimension...")

    score = 0  # Number of videos where all instances meet the requirement
    total_videos = 0  # Total number of videos processed

    for i, entry in tqdm(enumerate(dimension_data)):
        total_videos += 1  # Increment total videos
        sentence = entry["sentence"]
        instances = entry["instance"]
        video_name = entry["name"]

        video_file = os.path.join(video_folder_path, video_name)
        if not os.path.exists(video_file):
            logger.error(f"Video file {video_file} does not exist.")
            continue

        try:
            original_video, original_frame_time, original_video_time = load_video(video_file, 8, 1, force_sample=True)
        except Exception as e:
            logger.error(f"Failed to load video {video_file}: {e}")
            continue

        original_video, original_pre_instruction = build_video_prompt(
            image_processor, original_video, original_frame_time, original_video_time
        )

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], original_pre_instruction)
        conv.append_message(conv.roles[0], f"""Describe this video in one sentence, care about the {dimension_type}. 
                            Accurately capture every suspected character and try to guess the {dimension_type}.""")
        answer = get_answer(model, tokenizer, original_video, conv)

        logger.info(f"Video: {video_file}")
        logger.info(f"Sentence: {sentence}")
        logger.info(f"Generated Caption: {answer}")

        conv.append_message(conv.roles[1], answer)

        all_instances_meet = True  # Flag to check if all instances meet the requirement

        # Now process each instance
        for instance_id, instance in instances.items():
            instance_class = instance["class"]
            instance_specific = instance.get(dimension_type)

            tmp_conv = conv.copy()
            tmp_conv.append_message(
                tmp_conv.roles[0],
                f"Tell me if it is possible that '{instance_class}' is in the video?"
                f"Note the synonyms of '{instance_class}'. Your answer can only be YES or NO. "
                f"Do not output any answer that is not YES or NO."
            )
            answer_in_video = get_answer(model, tokenizer, original_video, tmp_conv)
            logger.info(f"Instance {instance_id}: {instance_class}")
            logger.info(f"In the video? {answer_in_video}")

            if not is_answer_yes_no(answer_in_video):
                all_instances_meet = False  # Instance not detected, video does not meet requirement
                logger.info(f"Instance {instance_id}: {instance_class} not detected in video.")
                break  # No need to check other instances

            conv_instance = conv.copy()
            conv_instance.append_message(
                conv_instance.roles[0],
                f"Based on your previous answer, tell me what is the '{dimension_type}' of '{instance_class}' in the video? "
                "Be careful to ignore camera movement."
            )
            answer_dimension = get_answer(model, tokenizer, original_video, conv_instance)
            conv_instance.append_message(conv_instance.roles[1], answer_dimension)

            logger.info(f"{dimension_type.capitalize()} (Ground Truth): {instance_specific}")
            logger.info(f"Model's {dimension_type.capitalize()} Answer: {answer_dimension}")

            if dimension_type == "detail":
                conv_instance.append_message(
                    conv_instance.roles[0],
                    f"Is the '{instance_specific}' of '{instance_class}' completely reflected in the video?"
                    "Your answer can only be YES or NO. Do not output any answer that is not YES or NO."
                )
            else:
                conv_instance.append_message(
                    conv_instance.roles[0],
                    f"Do you think the '{dimension_type}' of '{instance_class}' in the video completely matches to '{instance_specific}'? "
                    "Your answer can only be YES or NO. Do not output any answer that is not YES or NO."
                )

            final_answer = get_answer(model, tokenizer, original_video, conv_instance)
            conv_instance.append_message(conv_instance.roles[1], final_answer)

            logger.info(f"Answer: {final_answer}")
            if not is_answer_yes_no(final_answer):
                all_instances_meet = False  # Instance does not meet the requirement
                logger.info(f"Instance {instance_id}: {instance_class} does not meet the {dimension_type} requirement.")
                break  # No need to check other instances

            logger.info("*" * 60)

        if all_instances_meet:
            score += 1  # All instances in the video meet the requirement
            logger.info(f"Video {video_name} meets the {dimension_type} dimension requirements.")
        else:
            logger.info(f"Video {video_name} does NOT meet the {dimension_type} dimension requirements.")

        logger.info("=" * 60)

    if total_videos > 0:
        score_ratio = score / total_videos
    else:
        score_ratio = 0.0
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
            dimension_file_path = os.path.join("dimensions", "multiple", f"{dimension}.jsonl")
            video_folder_path = os.path.join(args.video_base, "multiple", dimension)

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
