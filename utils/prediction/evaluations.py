import cv2
import string
import pathlib
import numpy as np

from PIL import Image
from typing import List
from matplotlib import pyplot as plt

from utils.dataset import Dataset
from utils.rgb import rgb2mask

def visualize(save_path: pathlib.Path, prefix, **images):
    """
        Plot images in one row.
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        title_str = ' '.join(name.split('_'))
        title_str = title_str.replace('post', '%')
        title_str = title_str.title()

        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(title_str)
        plt.imshow(image, cmap='gray')

    if prefix is not None and save_path is not None:
        plt.savefig(f'{str(save_path.resolve())}/visualisation_{prefix}.png')
    else:
        plt.show()

def preload_image_data(data_dir: string, img_dir: string, is_mask: bool = False, patch_size: int = 256, dataset_list: string = 'test_dataset.txt'):
    """
        Loads all images from data_dir.
    """
    dataset_files: List = []
    dataset_file_names: List = []
    with open(pathlib.Path(data_dir, dataset_list), mode='r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            path = pathlib.Path(data_dir, img_dir, line.strip(), f'Image/{line.strip()}.png' if is_mask == False else f'Mask/0.png')
            img = None

            # Load image
            if is_mask:
                img = np.array(Image.open(str(path)).convert("L")) #cv2.imread(str(Path(info.mask, '0.png')), cv2.IMREAD_GRAYSCALE)
                img = Dataset._resize_and_pad(img, (patch_size, patch_size), (0, 0, 0))
            else:
                img = np.array(Image.open(str(path)).convert("RGB"))  #cv2.imread(str(Path(info.image, os.listdir(info.image)[0])))
                img = Dataset._resize_and_pad(img, (patch_size, patch_size), (0, 0, 0))

            dataset_files.append(img)
            dataset_file_names.append(line.strip())

    return dataset_files, dataset_file_names
