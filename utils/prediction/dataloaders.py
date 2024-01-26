"""
    Based on YOLOv5 dataloaders.py script.
    URL: https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py
"""

import os
import re
import cv2
import math
import glob
import time
import pathlib

import numpy as np

from threading import Thread
from utils.dataset import Dataset

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Constants
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'

# Utils
def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern='[|@#!¡·$€%&()=?¿^*;:,¨´><+]', repl='_', string=s)

# Image Loader
class LoadImages:
    def __init__(self, paths, img_size=256, transforms=None):
        if isinstance(paths, str) and pathlib.Path(paths).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            paths = pathlib.Path(paths).read_text().rsplit()

        files = []
        for path in sorted(paths) if isinstance(paths, (list, tuple)) else [paths]:
            path = pathlib.Path(path).resolve()
            path = fr'{str(path)}'

            if '*' in path:
                files.extend(sorted(glob.glob(path, recursive=True)))                                   # glob
            elif os.path.isdir(path):
                files.extend(sorted(glob.glob(os.path.join(path, '*.*'))))                              # Dir
            elif os.path.isfile(path):
                files.append(path)                                                                      # Files
            else:
                raise FileNotFoundError(f'{path} does not exist')

        images = [i for i in files if i.split('.')[-1].lower() in IMG_FORMATS]
        videos = [v for v in files if v.split('.')[-1].lower() in VID_FORMATS]
        num_images, num_videos = len(images), len(videos)
        
        self.img_size = img_size
        self.all_files = images + videos
        self.num_files = num_images + num_videos
        self.video_flag = [False] * num_images + [True] * num_videos
        self.transforms = transforms
        self.frame = 0
        self.frames = 0

        if any(videos):
            self._new_video(videos[0])
        else:
            self.cap = None

        if self.num_files == 0:
            log.error(f'No images nor videos found in {paths}')

    def __iter__(self):
        self.count = 0
        return self
    
    def skip_file(self):
        if self.count == self.num_files:
            raise StopIteration

        self.count += 1

    def __next__(self):
        if self.count == self.num_files:
            raise StopIteration

        path = self.all_files[self.count]
        if self.video_flag[self.count]:                                                                     # Read video
            self.cap.grab()
            ret_val, img_0 = self.cap.retrieve()

            while not ret_val:
                self.count += 1
                self.cap.release()

                if self.count == self.num_files:
                    raise StopIteration
                
                path = self.all_files[self.count]
                self._new_video(path)
                ret_value, img_0 = self.cap.read()

            img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
            img_0 = Dataset._resize_and_pad(img_0, (self.img_size, self.img_size), (0, 0, 0))
            self.frame += 1

        else:                                                                                               # Read image
            self.count += 1
            img_0 = cv2.imread(path)[:,:,::-1]
            img_0 = Dataset._resize_and_pad(img_0, (self.img_size, self.img_size), (0, 0, 0))

        img = img_0.astype(np.float32)
        img /= 255.0
        img = np.ascontiguousarray(img)

        if self.transforms:
            img = self.transforms(img)

        return path, img, img_0, self.cap, self.frame, self.frames

    def __len__(self):
        return self.num_files

    def _new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))

class LoadStreams:
    def __init__(self, source='', img_size=256, transforms=None):
        self.img_size = img_size
        self.transforms = transforms
        source = pathlib.Path(source).read_text().rsplit() if os.path.isfile(source) else source

        self.source = clean_str(source)
        self.imgs, self.fps, self.frames, self.threads = None, 0, 0, None
        stream = eval(source) if source.isnumeric() else source

        cap = cv2.VideoCapture(stream)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        self.frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30

        _, self.imgs = cap.read()
        self.threads = Thread(target=self.update, args=([cap, stream]), daemon=True)
        self.threads.start()            

    def update(self, cap, stream):
        frame_num, frame_arr = 0, self.frames
        while cap.isOpened() and frame_num < frame_arr:
            frame_num += 1
            cap.grab()
            ret_val, img_0 = cap.retrieve()

            if ret_val:
                self.imgs = img_0
            else:
                self.imgs = np.zeros_like(self.imgs)
                cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)

    def __iter__(self):
        self.count = -1
        return self
    
    def __next__(self):
        self.count += 1
        if not self.threads.is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        
        img_0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in img_0])
        else:
            img_0 = Dataset._resize_and_pad(img_0, (self.img_size, self.img_size), (0, 0, 0))
            img_0 = img_0[:,:,::-1]

            im = img_0.astype(np.float32)
            im /= 255.0
            im = np.ascontiguousarray(im)
        return self.source, im, img_0, None, 0, 0
    
    def __len__(self):
        return len(self.source)