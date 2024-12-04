import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re

# Function: Get the first frame of a video
def get_first_frame(video_path):
    try:
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            raise IOError("error")

        ret, first_frame = video_capture.read()
        video_capture.release()

        # If the first frame is successfully read, return it
        if ret:
            return first_frame
        else:
            raise IOError("Unable to read first frame")

    except Exception as e:
        print(f"error: {e}")
        # If an error occurs, return a 224x224 black image
        black_image = np.zeros((224, 224, 3), dtype=np.uint16)
        return black_image

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def sort_video_files(file_list):
    return sorted(file_list, key=natural_sort_key)

def get_and_sort_video_files(directory):
    video_files = [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.mp4')]
    return sort_video_files(video_files)

def get_and_sort_video_files_muti(video_files):
    return  sort_video_files(video_files)

# Custom Dataset class
class DETRDataset(Dataset):
    def __init__(self, video_folder, half='all', muti=False, transform=None):
        if muti:
            self.video_files = get_and_sort_video_files_muti(video_folder)
        else:
            self.video_files = get_and_sort_video_files(video_folder)
        if half=="first":
            self.video_files = self.video_files[len(self.video_files)//2:]
        elif half=="second":
            self.video_files = self.video_files[:len(self.video_files)//2]
        elif half=="all":
            self.video_files = self.video_files

        # List all video files with specific extensions
        self.transform = transform

    def __len__(self):
        # Return the total number of video files
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        # Get the first frame of the video
        first_frame = get_first_frame(video_path)

        # Apply image preprocessing if a transform is provided
        if self.transform:
            first_frame = self.transform(first_frame)

        return first_frame, video_path
