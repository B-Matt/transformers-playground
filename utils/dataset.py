import os
import cv2
import tqdm
import enum
import torch
import string
import pathlib
import numpy as np

from pathlib import Path
from multiprocessing.pool import ThreadPool
from os.path import splitext
from typing import List, Tuple, Any, Callable, Optional
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection

from tqdm import trange
from pycocotools.coco import COCO
from pycocotools import mask

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.general import data_info_tuple, NUM_THREADS

# Classes
class DatasetType(enum.Enum):
    TRAIN = 'training_dataset'
    VALIDATION = 'validation_dataset'
    TEST = 'test_dataset'

class DatasetCacheType(enum.Enum):
    NONE = 0,
    RAM = 1,
    DISK = 2,

class Dataset(Dataset):
    def __init__(self,
        data_dir: string,
        img_dir: string,
        images: List = None,
        cache_type: DatasetCacheType = DatasetCacheType.NONE,
        type: DatasetType = DatasetType.TRAIN,
        patch_size: int = 128,
        transform = None
    ) -> None:
        self.all_imgs = images
        self.is_searching_dirs = images == None and img_dir != None
        self.patch_size = patch_size
        self.transform = transform
        self.cache_type = cache_type
        self.images_data = []
        
        self.img_tupels = []
        if self.is_searching_dirs:
            self.img_tupels = self.preload_image_data_dir(data_dir, img_dir, type)
        else:
            self.img_tupels = self.preload_image_data(data_dir)

        if cache_type != DatasetCacheType.NONE:
            prefix = '[TRAINING]:' if type == DatasetType.TRAIN else '[VALIDATION]:'        
            fcn = self.cache_images_to_disk if cache_type == DatasetCacheType.DISK else self.load_sample
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(fcn, range(len(self.img_tupels)))
                pbar = tqdm.tqdm(enumerate(results), total=len(self.img_tupels))

                for i, x in pbar:
                    if cache_type == DatasetCacheType.RAM:
                        self.images_data.append(x)
                    pbar.desc = f'{prefix} Caching data'
                pbar.close()

    def preload_image_data(self, data_dir: string):
        dataset_files: List = []
        for image in self.all_imgs:
            data_info = data_info_tuple(
                image,
                pathlib.Path(data_dir, 'imgs', image),
                pathlib.Path(data_dir, 'masks', f'{splitext(image)[0]}_label.png')
            )
            dataset_files.append(data_info)
        return dataset_files

    def preload_image_data_dir(self, data_dir: string, img_dir: string, type: DatasetType):
        dataset_files: List = []
        with open(pathlib.Path(data_dir, f'{type.value}.txt'), mode='r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                path = pathlib.Path(data_dir, img_dir, line.strip())
                data_info = data_info_tuple(
                    line.strip(),
                    pathlib.Path(path, 'Image'),
                    pathlib.Path(path, 'Mask')
                )
                dataset_files.append(data_info)
        return dataset_files

    def load_sample(self, index):
        info = self.img_tupels[index]
        cache_path = Path('cache', info.name)
        if self.cache_type == DatasetCacheType.DISK and cache_path.exists():
            return np.load(cache_path)
        
        input_image = cv2.imread(str(Path(info.image, os.listdir(info.image)[0])), cv2.COLOR_BGR2RGB)
        input_image = input_image[:,:,::-1]
        input_image = Dataset._resize_and_pad(input_image, (self.patch_size, self.patch_size), (0, 0, 0))

        input_mask = cv2.imread(str(Path(info.mask, '0.png')), cv2.COLOR_BGR2RGB).astype('float32')
        input_mask = input_mask[:,:,::-1]
        input_mask = cv2.inRange(input_mask, (139, 189, 7), (139, 189, 7))
        input_mask = Dataset._resize_and_pad(input_mask, (self.patch_size, self.patch_size), (0, 0, 0))
        input_mask = input_mask.astype('float32')
        input_mask /= 255.0
        return input_image, input_mask

    def cache_images_to_disk(self, index):
        info = self.img_tupels[index]
        cache_path = Path('cache', info.name)
       
        if not cache_path.exists():
            img, mask = self.load_sample(index)
            np.savez(cache_path, img, mask)

    @staticmethod
    def _resize_and_pad(image: np.array, 
                    new_shape: Tuple[int, int],
                    padding_color: Tuple[int] = (0, 0, 0)
                ) -> np.array:
        """
        Maintains aspect ratio and resizes with padding.
        Params:
            image: Image to be resized.
            new_shape: Expected (width, height) of new image.
            padding_color: Tuple in BGR of padding color
        Returns:
            image: Resized image with padding
        """
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape)) / max(original_shape)
        new_size = tuple([int(x*ratio) for x in original_shape])
        image = cv2.resize(image, new_size)

        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        return image

    def __len__(self):
        return len(self.img_tupels)

    def __getitem__(self, index: int):
        if self.cache_type == DatasetCacheType.RAM:
            img, mask = self.images_data[index]
        else:
            img, mask = self.load_sample(index)

        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            temp_img = augmentation['image']
            temp_mask = augmentation['mask']

        temp_mask = temp_mask.unsqueeze(0)

        return {
            'image': torch.as_tensor(temp_img).float(),
            'mask': torch.as_tensor(temp_mask).float()
        }

class BinaryDataset(Dataset):
    def __init__(self,
        data_dir: string,
        img_dir: string,
        images: List = None,
        cache_type: DatasetCacheType = DatasetCacheType.NONE,
        type: DatasetType = DatasetType.TRAIN,
        patch_size: int = 128,
        transform = None
    ) -> None:
        self.all_imgs = images
        self.is_searching_dirs = images == None and img_dir != None
        self.patch_size = patch_size
        self.transform = transform
        self.cache_type = cache_type
        self.images_data = []
        
        self.img_tupels = []
        if self.is_searching_dirs:
            self.img_tupels = self.preload_image_data_dir(data_dir, img_dir, type)
        else:
            self.img_tupels = self.preload_image_data(data_dir)
    
    def preload_image_data(self, data_dir: string):
        dataset_files: List = []
        for image in self.all_imgs:
            data_info = data_info_tuple(
                image,
                pathlib.Path(data_dir, 'imgs', image),
                pathlib.Path(data_dir, 'masks', f'{splitext(image)[0]}_label.png')
            )
            dataset_files.append(data_info)
        return dataset_files

    def preload_image_data_dir(self, data_dir: string, img_dir: string, type: DatasetType):
        dataset_files: List = []
        with open(pathlib.Path(data_dir, f'{type.value}.txt'), mode='r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                path = pathlib.Path(data_dir, img_dir, line.strip())
                data_info = data_info_tuple(
                    line.strip(),
                    pathlib.Path(path, 'Image'),
                    pathlib.Path(path, 'Mask')
                )
                dataset_files.append(data_info)
        return dataset_files

    def load_sample(self, index):
        info = self.img_tupels[index]
        cache_path = Path('cache', info.name)
        if self.cache_type == DatasetCacheType.DISK and cache_path.exists():
            return np.load(cache_path)
        
        input_image = np.array(Image.open(str(Path(info.image, os.listdir(info.image)[0]))).convert("RGB"))  #cv2.imread(str(Path(info.image, os.listdir(info.image)[0])))
        input_image = Dataset._resize_and_pad(input_image, (self.patch_size, self.patch_size), (0, 0, 0))

        input_mask = np.array(Image.open(str(Path(info.mask, '0.png'))).convert("L")) #cv2.imread(str(Path(info.mask, '0.png')), cv2.IMREAD_GRAYSCALE)
        input_mask = Dataset._resize_and_pad(input_mask, (self.patch_size, self.patch_size), (0, 0, 0))
        return input_image, input_mask

    def __len__(self):
        return len(self.img_tupels)
    
    def __getitem__(self, index: int):
        img, mask = self.load_sample(index)

        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            temp_img = augmentation['image']
            temp_mask = augmentation['mask']

        temp_mask = temp_mask / 255.0
        temp_mask = temp_mask.unsqueeze(0)

        return {
            'image': torch.as_tensor(temp_img).float(),
            'mask': temp_mask
        }
    
class CocoDataset(CocoDetection):
    """
        Rewritten CocoDetector from PyTorch: `https://pytorch.org/vision/main/_modules/torchvision/datasets/coco.html`
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        patch_size: int = 128,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, None, None, None)

        self.patch_size = patch_size
        self.transform = transform

    def load_sample(self, id):        
        input_image = self.coco.loadImgs(id)[0]["file_name"]
        input_image = Dataset._resize_and_pad(input_image, (self.patch_size, self.patch_size), (0, 0, 0))

        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        input_mask = self.coco.annToMask(anns[0])
        for i in (range(len(anns))):
            input_mask = input_mask | self.coco.annToMask(anns[i])

        input_mask = Dataset._resize_and_pad(input_mask, (self.patch_size, self.patch_size), (0, 0, 0))
        input_mask = input_mask.astype('float32')
        return input_image, input_mask
    
    def _get_mask(self, anns):
        if len(anns) == 0:
            return np.zeros((self.patch_size, self.patch_size))
    
        mask = self.coco.annToMask(anns[0])
        for i in (range(len(anns))):
            mask = mask | self.coco.annToMask(anns[i])

        mask = Dataset._resize_and_pad(mask, (self.patch_size, self.patch_size), (0, 0, 0))
        mask = mask.astype('float32')
        return mask

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target_anns = super().__getitem__(index)
        img = Dataset._resize_and_pad(np.array(img), (self.patch_size, self.patch_size), (0, 0, 0))
        mask = self._get_mask(target_anns)

        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            temp_img = augmentation['image']
            temp_mask = augmentation['mask']

        #temp_mask = temp_mask.unsqueeze(0)
        return {
            'image': torch.as_tensor(temp_img).float(),
            'mask': torch.as_tensor(temp_mask).float()
        }

    def __len__(self) -> int:
        return len(self.ids)

class COCOSegmentation(Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    def __init__(
            self,
            idsFile = '',
            imgDir = '',
            annFile = '',
            patch_size: int = 128,
            transform = None
        ):
        self.patch_size = patch_size
        self.img_dir = imgDir
        
        self.coco = COCO(annFile)
        self.coco_mask = mask
        self.transform = transform

        if os.path.exists(idsFile):
            self.ids = torch.load(idsFile)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, idsFile)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        if self.transform is not None:
            augmentation = self.transform(image=_img, mask=_target)
            temp_img = augmentation['image']
            temp_mask = augmentation['mask']

        return {
            'image': torch.as_tensor(temp_img).float(),
            'mask': torch.as_tensor(temp_mask).float()
        }

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']

        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')        
        _img = Dataset._resize_and_pad(np.array(_img), (self.patch_size, self.patch_size), (0, 0, 0))

        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])
        _target = Dataset._resize_and_pad(_target, (self.patch_size, self.patch_size), (0, 0, 0))
        return _img, _target

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def __len__(self):
        return len(self.ids)