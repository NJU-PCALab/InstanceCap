# Originally developed by https://github.com/Vchitect/VBench based on https://github.com/facebookresearch/co-tracker.

import argparse
from typing import List
import json

from .camera_motion import compute_camera_motion


def process(paths: List[str], threshold: float) -> List[str]:
    device = "cuda"
    print("loading model...")
    submodules = {"repo": "facebookresearch/co-tracker", "model": "cotracker2"}
    camera_motion_types = compute_camera_motion(device, submodules, paths, factor=threshold)
    return camera_motion_types


def main(args):
    video_list = []
    questions_list = []
    with open(args.questions_path, 'r', encoding='utf-8') as file:
        for line in file:
            video_list.append(json.loads(line.strip())['video_path'])
            questions_list.append(json.loads(line.strip()))

    results = process(video_list, args.threshold)
    for i in range(len(questions_list)):
        question = questions_list[i]
        question['Camera Movement'] = results[i]

    with open(args.questions_path, 'w', encoding='utf-8') as file:
        for question in questions_list:
            file.write(json.dumps(question, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-path", type=str, default="../questions_1.jsonl")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    main(args)
