import argparse
from typing import List
import json
import os
import multiprocessing
from .camera_motion import compute_camera_motion  # 请确保正确导入模块


def process(questions_chunk: List[dict], threshold: float, device: str, output_file: str):
    print(f"Process on device {device}: loading model...")
    submodules = {"repo": "facebookresearch/co-tracker", "model": "cotracker2"}
    paths = [q['video_path'] for q in questions_chunk]
    camera_motion_types = compute_camera_motion(device, submodules, paths, factor=threshold)

    # 更新问题列表，添加 'Camera Movement' 字段
    for i, question in enumerate(questions_chunk):
        question['Camera Movement'] = camera_motion_types[i]

    # 将结果写入临时文件
    with open(output_file, 'w', encoding='utf-8') as file:
        for question in questions_chunk:
            file.write(json.dumps(question, ensure_ascii=False) + '\n')


def main(args):
    multiprocessing.set_start_method('spawn')
    video_list = []
    questions_list = []
    with open(args.questions_path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            question = json.loads(line.strip())
            question['index'] = idx  # 添加索引以保持顺序
            video_list.append(question['video_path'])
            questions_list.append(question)

    # 将问题列表分成四个子列表
    num_processes = 4  # 使用四张GPU
    chunks = [[] for _ in range(num_processes)]
    for i, question in enumerate(questions_list):
        chunks[i % num_processes].append(question)

    processes = []
    temp_files = []
    for rank in range(num_processes):
        device = f'cuda:{rank}'
        temp_file = f"{args.questions_path}_temp_{rank}.jsonl"
        temp_files.append(temp_file)
        p = multiprocessing.Process(target=process, args=(chunks[rank], args.threshold, device, temp_file))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 收集所有结果，按照索引排序，移除索引，写回到原始文件
    combined_questions = []
    for temp_file in temp_files:
        with open(temp_file, 'r', encoding='utf-8') as file:
            for line in file:
                question = json.loads(line.strip())
                combined_questions.append(question)
        os.remove(temp_file)  # 删除临时文件

    # 按照索引排序
    combined_questions.sort(key=lambda x: x['index'])

    # 移除索引字段
    for question in combined_questions:
        del question['index']

    # 写回到原始文件
    with open(args.questions_path, 'w', encoding='utf-8') as file:
        for question in combined_questions:
            file.write(json.dumps(question, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-path", type=str, default="../questions(instance).jsonl")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    main(args)
