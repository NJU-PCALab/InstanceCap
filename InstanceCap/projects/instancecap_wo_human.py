import argparse

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
import av
import torch
from aiohttp.web_routedef import static
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import numpy as np
import json
from tqdm import tqdm
from video2ins import get_blue_screen_videos, get_gaussian_blur_videos
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
from human_hit import category_prompts

device = "cuda"

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames,frame_time,video_time

def extract_substring_after_last_occurrence(output, s):
    last_index = output.rfind(s)
    if last_index == -1:
        return output
    start_index = last_index + len(s)
    return output[start_index:]

def read_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def build_video_prompt(image_processor, video, frame_time, video_time):
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. "\
                        f"These frames are located at {frame_time}.Please answer the following questions related to this video."
    pre_instruction = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n"

    return video, pre_instruction

def get_answer(model, tokenizer, video, conv):
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    text_outputs = extract_substring_after_last_occurrence(text_outputs, "assistant\n")
    return text_outputs

def interfere(model, tokenizer, image_processor, predictor, question):
    video_path = question['video_path']
    conv_template = "qwen_1_5"
    system_prompt = read_prompt("LLaVA-NeXT/system_prompt.txt")
    max_frames_num = 8
    original_video, original_frame_time, original_video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    original_video, original_pre_instruction = build_video_prompt(image_processor, original_video, original_frame_time, original_video_time)

    general_conv = copy.deepcopy(conv_templates[conv_template])
    general_conv.append_message(general_conv.roles[0], original_pre_instruction)
    general_conv.append_message(general_conv.roles[0], "Please describe this video in one sentence, no more than 20 words.\n")
    general_conv.append_message(general_conv.roles[1], None)
    general_description = get_answer(model, tokenizer, original_video, general_conv)

    dict_ = {}
    items = ["Main Instance", "Background Detail", "Camera Movement"]
    characters = ["Appearance", "Actions and Motion"]

    if question["target_objects"]:
        videos, positions, frame_time, video_time = get_gaussian_blur_videos(question, predictor, num_frames=8)

        new_videos = []
        for i in range(len(videos)):
            video, pre_instruction = build_video_prompt(image_processor, videos[i],
                                                                      frame_time, video_time)
            new_videos.append(video)

    for item in items:
        if item == "Main Instance":
            dict_[item] = {}
            objects = question["target_objects"]
            if not objects:
                continue

            for i in range(len(objects)):
                dict_[item]["No." + str(i)] = {}
                target_object = objects[i]
                dict_[item]["No." + str(i)]["Class"] = target_object

                for character in characters:
                    if character == "Appearance":
                        target_conv = copy.deepcopy(conv_templates[conv_template])
                        target_conv.system = f"""<|im_start|>system
                        {system_prompt}"""
                        target_conv.append_message(target_conv.roles[0], pre_instruction)

                        for round in range(2):
                            if round == 0:  # CoT
                                target_conv.append_message(target_conv.roles[0], "Let's think step by step...\n")
                                target_conv.append_message(target_conv.roles[0],
                                                           f"Can you tell what color the background of this video is? "
                                                           f"Unimportant parts have been intentionally obscured with a mosaic; please focus only on the clearly visible portions. If I remind you that there's a {target_object}, can you see it in the video?\n"
                                                           f"Continue ignore mosaic background, describe the appearance and especially clearly visible text of the {target_object} in video you see.\n"
                                                           )

                                answer = get_answer(model, tokenizer, new_videos[i], target_conv)
                                target_conv.append_message(target_conv.roles[1], answer + '\n')

                            if round == 1:
                                target_conv.append_message(target_conv.roles[0],
                                                                         f"Do not answer any information unrelated to {target_object}'s appearance or you're not sure. Give description in detail.\n"
                                                                         f"Do not answer in bullet points. Do not mention anything about a \"blurred background.\"\n"
                                                                         f"Do not speculate about the environment in which the {target_object} is located.\n")

                                target_conv.append_message(target_conv.roles[0],
                                                           category_prompts[target_object] + "\n")

                                answer = get_answer(model, tokenizer, new_videos[i], target_conv)
                                dict_[item]["No." + str(i)][character] = answer

                    else:
                        target_conv = copy.deepcopy(conv_templates[conv_template])
                        target_conv.system = f"""<|im_start|>system
                        {system_prompt}"""
                        target_conv.append_message(target_conv.roles[0], pre_instruction)

                        for round in range(2):
                            if round == 0:  # CoT
                                target_conv.append_message(target_conv.roles[0], "Let's think step by step...\n")
                                target_conv.append_message(target_conv.roles[0],
                                                           f"Can you tell what color the background of this video is? Unimportant parts have been intentionally obscured with a mosaic; please focus only on the clearly visible portions. If I remind you that there's a {target_object}, can you see it in the video?\n"
                                                           f"This is supplementary information to the full video to aid your description:\n{general_description}\n"
                                                           f"Can you find the {target_object} corresponding to this paragraph in the video?\n")

                                answer = get_answer(model, tokenizer, new_videos[i], target_conv)
                                target_conv.append_message(target_conv.roles[1], answer + '\n')

                            if round == 1:
                                target_conv.append_message(target_conv.roles[0],
                                                                         f"Continue ignore mosaic background, if {target_object} has any movement, answer what you think {target_object} is doing in video.\n"
                                                                         f"Extract the information related to {target_object} that you are currently describing.\n"
                                                                         f"Do not answer any information unrelated to {target_object}'s state of motion or you're not sure. Observe carefully, use appropriate adjectives, and give description in detail.\n"
                                                                         f"Do not answer in bullet points. Do not mention any objects that are not related to {target_object}.\n"
                                                                         f"Do not speculate about the environment in which the {target_object} is located.\n"
                                                                         f"Do not mention anything about a \"blurred background.\" Use a continuous paragraph.\n")

                                answer = get_answer(model, tokenizer, new_videos[i], target_conv)
                                dict_[item]["No." + str(i)][character] = answer

                # Stay at A
                if positions[i][0] == positions[i][1]:
                    answer = f"{target_object} stays at {positions[i][0]}"
                else:
                    # From A to B
                    answer = f"{target_object} moves from {positions[i][0]} to {positions[i][1]}"

                # out of frame
                answer += " and out of frame in the end." if positions[i][2] else "."
                dict_[item]["No." + str(i)]["Position"] = answer

        if item == "Background Detail":
            background_conv = copy.deepcopy(conv_templates[conv_template])
            background_conv.system = f"""<|im_start|>system
            {system_prompt}"""
            background_conv.append_message(background_conv.roles[0], original_pre_instruction)

            for round in range(2):
                if round == 0:  # CoT
                    background_conv.append_message(background_conv.roles[0], "Let's think step by step...\n")
                    background_conv.append_message(background_conv.roles[0],
                                               f"Can you tell which foreground exist in this video? If you ignore these foreground, can you tell me what environment look like?\n")

                    answer = get_answer(model, tokenizer, original_video, background_conv)
                    background_conv.append_message(background_conv.roles[1], answer + '\n')

                if round == 1:
                    background_conv.append_message(background_conv.roles[0],
                                                   f"Please continue ignore the foreground you just mentioned and just describe the environment looks like in detail.\n"
                                                   f"Please note, do not analyze, just describe what you see. Observe carefully, use appropriate adjectives, and give description in detail.\n"
                                                   f"Do not answer in bullet points.\n")

                    answer = get_answer(model, tokenizer, original_video, background_conv)
                    dict_[item] = answer

        if item == "Camera Movement":
            camera_motion = question["Camera Movement"]
            camera_conv = copy.deepcopy(conv_templates[conv_template])
            camera_conv.system = f"""<|im_start|>system
            {system_prompt}"""
            camera_conv.append_message(camera_conv.roles[0], original_pre_instruction)

            camera_conv.append_message(camera_conv.roles[0], "Let's think step by step...\n"
                                                             "Try to separate the camera movement from the video.\n")

            if camera_motion == "Undetermined":
                camera_conv.append_message(camera_conv.roles[0],
                                           "The motion of the video camera is very complex, can you infer the possible motion of the camera and the shooting Angle (long distance/medium distance/overhead Angle/POV, etc.) from the changes in the video?")
            elif camera_motion == "static":
                camera_conv.append_message(camera_conv.roles[0],
                                           "Is the camera static or moving? Can you deduce the possible motion of the camera and the shooting Angle (long distance/medium distance/overhead Angle/POV, etc.) in the video?\n")
            else:
                camera_conv.append_message(camera_conv.roles[0],
                                           f"The motion of the camera in this video is {camera_motion}. According to my tips, can you deduce the possible motion of the camera and the shooting Angle (long distance/medium distance/overhead Angle/POV, etc.) in the video?")


            camera_conv.append_message(camera_conv.roles[0],
                                       f"Summarize the camera movement and shooting angle, use degree adverbs appropriately(Sharply, rapidly, slowly, etc), try to give description in detail.")
            answer = get_answer(model, tokenizer, original_video, camera_conv)
            dict_[item] = answer

    return dict_, general_description

def main(args):
    # load llava-next-video
    pretrained = args.llava_next_video_path
    model_name = "llava_qwen"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,
                                                                          torch_dtype="bfloat16", device_map=device_map,
                                                                          ignore_mismatched_sizes=True)  # Add any other thing you want to pass in llava_model_args
    model.eval().type(torch.bfloat16)

    # load sam
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    from sam2.build_sam import build_sam2_video_predictor

    # load model
    sam2_checkpoint = args.sam_path
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    question_list = []
    with open(args.questions_path, 'r', encoding='utf-8') as file:
        for line in file:
            question_list.append(json.loads(line.strip()))

    output_file_path = args.output_path

    # Handle --half parameter
    if args.half == 'first':
        question_list = question_list[:len(question_list) // 2]
    elif args.half == 'second':
        question_list = question_list[len(question_list) // 2:]

    print(len(question_list))
    with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
        for question in tqdm(question_list, desc="Processing"):
            answer = {"Video": question["video_path"]}
            dict_, general_description = interfere(model, tokenizer, image_processor, predictor, question)
            answer["Global Description"] = general_description
            answer["Structural Description"] = dict_
            json_str = json.dumps(answer, ensure_ascii=False)
            jsonl_file.write(json_str + '\n')
            print(json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava-next-video-path", type=str, default="/home/weights/LLaVA-Video-72B-Qwen2")
    parser.add_argument("--questions-path", type=str,
                        default="../questions_1.jsonl")
    parser.add_argument("--sam-path", type=str,
                        default="./segment-anything-2/checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--output-path", type=str, default="../1.jsonl")
    parser.add_argument("--half", type=str, choices=['first', 'second', 'all'], default='all',
                        help="Process first half, second half or all of the data")

    args = parser.parse_args()

    main(args)
