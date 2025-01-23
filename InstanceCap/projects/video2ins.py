import os
import sys

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
import cv2
import shutil

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
    # import pdb;pdb.set_trace()
    return spare_frames, frame_time, video_time

def video_to_frames(video_path, output_dir="../temp", extension="jpg", num_frames=None):
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    spare_frames, frame_time, video_time = load_video(video_path, num_frames)
    for i, frame in enumerate(spare_frames):
        filename = f"{i:05d}.{extension}"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
    return spare_frames, frame_time, video_time

def get_blur_frames(frame):
    h, w, _ = frame.shape
    frame = cv2.resize(frame, (max(w//150, 1), max(h//150, 1)))
    frame = cv2.resize(frame, (w, h))

    return frame

def apply_gaussian_blur(mask, image, kernel_size=(81, 81)):
    scenic_mask = ~mask
    scenic_mask = np.repeat(scenic_mask[0, ...][..., None], 3, axis=-1)

    blurred = get_blur_frames(image)
    blurred_masked = np.where(scenic_mask, blurred, image)

    return blurred_masked

def add_blue_screen(mask, image):
    blue_color = np.array([255, 0, 0], dtype=np.uint8)
    h, w = mask.shape[-2:]
    blue_screen = np.ones((h, w, 3), dtype=np.uint8) * blue_color

    scenic_mask = ~mask

    image = image * np.repeat(mask[..., None], 3, axis=-1)
    blue_screen = blue_screen * np.repeat(scenic_mask[..., None], 3, axis=-1)

    return cv2.add(image, blue_screen)[0]

def get_region(center):
    """
    Given a point's normalized coordinates (both between 0 and 1),
    this function returns the name of the region where the point lies.
    The image/space is divided into a 3x3 grid:
    Top left, top middle, top right
    Center left, center middle, center right
    Bottom left, bottom middle, bottom right

    Parameters:
    - x: float, the horizontal coordinate of the point (between 0 and 1).
    - y: float, the vertical coordinate of the point (between 0 and 1).

    Returns:
    A string representing the region name.
    """
    if center == None:
        return "out of frame"
    x, y = center
    if x < 1 / 3:
        x_position = 'left'
    elif x < 2 / 3:
        x_position = 'middle'
    else:
        x_position = 'right'

    if y > 1 / 3:
        y_position = 'bottom'
    elif y < 2 / 3:
        y_position = 'center'
    else:
        y_position = 'top'

    return f'{y_position}-{x_position}'

def find_center_of_trues(matrix):
    """
    Finds the center of all True values in a boolean matrix and returns its normalized coordinates.

    Parameters:
    - matrix: A 2D numpy array of boolean values.

    Returns:
    A tuple representing the normalized coordinates (x, y) of the center of True values.
    If there are no True values, returns None.
    """
    # Find the indices of all True values
    true_indices = np.argwhere(matrix)

    if len(true_indices) == 0:
        return None  # No True values found

    # Calculate the mean position of all True values
    center_y, center_x = np.mean(true_indices, axis=0)

    # Normalize the coordinates by the size of the matrix
    height, width = matrix.shape
    norm_center_x = center_x / width
    norm_center_y = center_y / height

    return (norm_center_x, norm_center_y)

def process_list(lst):
    """
    Processes a list of strings and returns the first element, the first non-"out of frame" element from the end,
    and a boolean indicating whether the last element is "out of frame".

    Parameters:
    - lst: List[str], a list of strings.

    Returns:
    A tuple containing:
    - The first element of the list.
    - The first non-"out of frame" element from the end of the list or None if all elements are "out of frame".
    - A boolean indicating if the last element is "out of frame".
    """
    # First element
    first_element = lst[0] if lst else None

    # First non-"out of frame" element from the end
    reversed_lst = reversed(lst)
    last_non_out_of_frame = next((item for item in reversed_lst if item != "out of frame"), None)

    # Whether the last element is "out of frame"
    last_element_is_out_of_frame = lst[-1] == "out of frame" if lst else False

    return first_element, last_non_out_of_frame, last_element_is_out_of_frame

def frames_to_video(frames, output_path, fps=23):
    # 获取第一帧的尺寸
    frame_shape = frames[0].shape[:2]
    height, width = frame_shape

    # 初始化 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 写入每一帧
    for frame in frames:
        # 确保帧的数据类型为 uint8，这是最常见的图像数据类型
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # 写入帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)

    # 释放 VideoWriter
    out.release()

def get_gaussian_blur_videos(question, predictor, video_dir = "../temp", num_frames=8):
    bboxes = question['bboxes']
    spare_frames, frame_time, video_time = video_to_frames(question['video_path'], output_dir=video_dir, num_frames=num_frames)
    frame_count = len(spare_frames)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_names = frame_names[:frame_count]
    frames = [cv2.imread(os.path.join(video_dir, p)) for p in frame_names]

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    for ann_obj_id, bbox in enumerate(bboxes):
        box = np.array(bbox)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=ann_obj_id,
            box=box,
        )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    positions = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    blur_frames = {}
    for out_frame_idx in tqdm(range(0, len(frame_names)), desc='Add blue screen'):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            if out_obj_id not in blur_frames:
                blur_frames[out_obj_id] = []
                positions[out_obj_id] = []

            image = frames[out_frame_idx]
            blur_frames[out_obj_id].append(apply_gaussian_blur(out_mask, image))

            center = find_center_of_trues(out_mask[0])
            positions[out_obj_id].append(get_region(center))

    for obj_id in blur_frames:
        blur_frames[obj_id] = np.array(blur_frames[obj_id])
        positions[obj_id] = process_list(positions[obj_id])

    # for i in range(len(blur_frames)):
    #     print(f"output {i}")
    #     frames_to_video(blur_frames[i], f'../videos/blur_frames{i}.mp4', fps=2)

    return blur_frames, positions, frame_time, video_time

def get_blue_screen_videos(question, predictor, video_dir = "../temp", num_frames=8):
    bboxes = question['bboxes']
    spare_frames, frame_time, video_time = video_to_frames(question['video_path'], output_dir=video_dir, num_frames=num_frames)
    frame_count = len(spare_frames)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_names = frame_names[:frame_count]
    frames = [cv2.imread(os.path.join(video_dir, p)) for p in frame_names]

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    for ann_obj_id, bbox in enumerate(bboxes):
        box = np.array(bbox)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=ann_obj_id,
            box=box,
        )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    positions = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    blue_screen_frames = {}
    for out_frame_idx in tqdm(range(0, len(frame_names)), desc='Add blue screen'):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            if out_obj_id not in blue_screen_frames:
                blue_screen_frames[out_obj_id] = []
                positions[out_obj_id] = []

            image = frames[out_frame_idx]
            blue_screen_frames[out_obj_id].append(add_blue_screen(out_mask, image))

            center = find_center_of_trues(out_mask[0])
            positions[out_obj_id].append(get_region(center))

    for obj_id in blue_screen_frames:
        blue_screen_frames[obj_id] = np.array(blue_screen_frames[obj_id])
        positions[obj_id] = process_list(positions[obj_id])

    return blue_screen_frames, positions, frame_time, video_time


def apply_ellipse_red_circle(mask, image):
    # 找到mask的中心点
    center = find_center_of_trues(mask[0])

    # 计算椭圆的轴长
    axes_length = (50, 30)  # 你可以根据需要调整椭圆的大小

    # 在图像上绘制椭圆红圈
    cv2.ellipse(image, center, axes_length, 0, 0, 360, (0, 0, 255), 2)  # 红色椭圆的厚度为2

    return image


def get_ellipse_red_circle_videos(question, predictor, video_dir="../temp", num_frames=8):
    bboxes = question['bboxes']
    spare_frames, frame_time, video_time = video_to_frames(question['video_path'], output_dir=video_dir,
                                                           num_frames=num_frames)
    frame_count = len(spare_frames)

    # 扫描目录中的所有JPEG帧
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_names = frame_names[:frame_count]
    frames = [cv2.imread(os.path.join(video_dir, p)) for p in frame_names]

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    for ann_obj_id, bbox in enumerate(bboxes):
        box = np.array(bbox)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=ann_obj_id,
            box=box,
        )

    video_segments = {}  # video_segments包含每帧的分割结果
    positions = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    ellipse_frames = {}
    for out_frame_idx in tqdm(range(0, len(frame_names)), desc='Add red ellipse'):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            if out_obj_id not in ellipse_frames:
                ellipse_frames[out_obj_id] = []
                positions[out_obj_id] = []

            image = frames[out_frame_idx]
            ellipse_frames[out_obj_id].append(apply_ellipse_red_circle(out_mask, image))

            center = find_center_of_trues(out_mask[0])
            positions[out_obj_id].append(get_region(center))

    for obj_id in ellipse_frames:
        ellipse_frames[obj_id] = np.array(ellipse_frames[obj_id])
        positions[obj_id] = process_list(positions[obj_id])

    return ellipse_frames, positions, frame_time, video_time
