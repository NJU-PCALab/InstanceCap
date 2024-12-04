from mmdet.apis import DetInferencer
import argparse
import os
from tqdm import tqdm
import cv2
import numpy as np
import json
import re
from video_dataloader import DETRDataset
from torch.utils.data import DataLoader
import multiprocessing
import torch

classes = np.array(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])

def process_bboxes(bboxes):
    bboxes_info = []

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox

        width = x_max - x_min
        height = y_max - y_min
        area = width * height

        bboxes_info.append([x_min, y_min, x_max, y_max, area])

    bboxes_info = np.array(bboxes_info).astype(np.uint32)

    return bboxes_info

def top_indices_3d(array, nums=3):
    sorted_indices = np.argsort(array[:, -1])[::-1]
    num_to_return = min(nums, len(sorted_indices))
    return sorted_indices[:num_to_return]

def get_first_frame(video_path):
    try:
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            raise IOError("error")

        ret, first_frame = video_capture.read()
        video_capture.release()

        return first_frame

    except Exception as e:
        print(f"error：{e}")
        black_image = np.zeros((256, 256, 3), dtype=np.uint16)
        return black_image

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def sort_video_files(file_list):
    return sorted(file_list, key=natural_sort_key)

def get_and_sort_video_files(directory):
    video_files = [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.mp4')]
    return sort_video_files(video_files)

def custom_collate_fn(batch):
    # Unpack frames and paths and return them as lists
    frames, paths = zip(*batch)
    return list(frames), list(paths)

def get_json(results, video_path, args):
    dict_ = {}
    labels = []
    bboxes = []

    for i in range(len(results['scores'])):
        if results['scores'][i] <= args.threshold:
            break
        labels.append(results['labels'][i])
        bboxes.append(results['bboxes'][i])

    # 如果labels为空
    if not labels:
        dict_["video_path"] = video_path
        dict_["target_objects"] = []
        dict_["bboxes"] = []
    else:
        labels = classes[np.array(labels).astype(np.int32)]
        bboxes = process_bboxes(bboxes)
        index = top_indices_3d(bboxes, nums=args.nums)

        dict_["video_path"] = video_path
        dict_["target_objects"] = labels[index].tolist()
        dict_["bboxes"] = bboxes[index, :-1].tolist()

    return dict_

def process_videos(video_files, rank, args):
    # 设置设备
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    # 初始化推理器
    inferencer = DetInferencer(model=args.model_path, weights=args.weights_path, device=device)

    # 创建数据集和数据加载器
    dataset = DETRDataset(video_files, muti=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    # 设置输出文件路径
    output_file_path = f"{args.output_path}_rank{rank}.jsonl"

    with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
        for frames, video_paths in tqdm(dataloader, desc=f"Process {rank}", position=rank):
            results = inferencer(frames, batch_size=len(frames))['predictions']
            for i in range(len(results)):
                dict_ = get_json(results[i], video_paths[i], args)
                json_str = json.dumps(dict_, ensure_ascii=False)
                jsonl_file.write(json_str + '\n')
                # 打印信息时注明进程编号
                print(f"Process {rank}: {json_str}")

def main(args):
    multiprocessing.set_start_method('spawn')
    video_files = get_and_sort_video_files(args.video_dir)

    num_processes = 4  
    video_file_splits = np.array_split(video_files, num_processes)

    processes = []
    for rank in range(num_processes):
        p = multiprocessing.Process(target=process_videos, args=(video_file_splits[rank], rank, args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    with open(args.output_path, 'w', encoding='utf-8') as outfile:
        for rank in range(num_processes):
            output_file_path = f"{args.output_path}_rank{rank}.jsonl"
            with open(output_file_path, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            os.remove(output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="./mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py")
    parser.add_argument("--weights-path", type=str,
                        default="./mmdetection/projects/CO-DETR/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth")
    parser.add_argument("--video-dir", type=str, default="/home/caption/1video/")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-path", type=str, default="../questions_1.jsonl")
    parser.add_argument("--nums", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    main(args)
