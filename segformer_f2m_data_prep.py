import os
import cv2
import tqdm
import torch
import pathlib

import albumentations as A
import numpy as np

from PIL import Image
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation

from utils.prediction.evaluations import preload_image_data

## Vars

id2label = { 0: 'background', 1: 'fire' }
label2id = { 'background': 0, 'fire': 1 }

model_checkpoint = "nvidia/mit-b3"

models_data = [
    {
        'model_name': 'SegFormer-256x256',
        'model_path': r'checkpoints/segformer/scarlet-totem-58',
        'patch_size': 256,
    },
    {
        'model_name': 'SegFormer-640x640',
        'model_path': r'checkpoints/segformer/sandy-resonance-59',
        'patch_size': 640,
    },
    {
        'model_name': 'SegFormer-800x800',
        'model_path': r'checkpoints/segformer/morning-moon-60',
        'patch_size': 800,
    },
]

val_transforms = A.Compose(
    [
        ToTensorV2(),
    ],
)

# Functions
def prepare_directory(model_data, dataset_type):
    preped_data = pathlib.Path('preped_data')
    model_output = pathlib.Path(preped_data, model_data['model_name'], dataset_type)

    # Create directory if it doesn't exists
    if not os.path.isdir(model_output):
        os.makedirs(model_output)

def get_image_directory(model_data, dataset_type):
    preped_data = pathlib.Path('preped_data')
    model_output = pathlib.Path(preped_data, model_data['model_name'], dataset_type)
    return model_output

def _resize_and_pad(image: np.array, 
                    new_shape,
                    padding_color = (0, 0, 0)
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

# def preload_image_data(data_dir, img_dir, is_mask: bool = False, patch_size: int = 256, dataset_list = 'test_dataset.txt'):
#     """
#         Loads all images from data_dir.
#     """
#     dataset_files = []
#     dataset_file_names = []
#     with open(pathlib.Path('/media/matej/B4085A48085A09AE/Posao/Transformers-Playground', data_dir, dataset_list), mode='r', encoding='utf-8') as file:
#         for i, line in enumerate(file):
#             path = pathlib.Path('/media/matej/B4085A48085A09AE/Posao/Transformers-Playground', data_dir, img_dir, line.strip(), f'Image/{line.strip()}.png' if is_mask == False else f'Mask/0.png')
#             img = None

#             # Load image
#             if is_mask:
#                 img = np.array(Image.open(str(path)).convert("L")) #cv2.imread(str(Path(info.mask, '0.png')), cv2.IMREAD_GRAYSCALE)
#                 img = _resize_and_pad(img, (patch_size, patch_size), (0, 0, 0)).astype(np.float32)
#             else:
#                 img = np.array(Image.open(str(path)).convert("RGB"))  #cv2.imread(str(Path(info.image, os.listdir(info.image)[0])))
#                 img = _resize_and_pad(img, (patch_size, patch_size), (0, 0, 0)).astype(np.float32)

#             dataset_files.append(img)
#             dataset_file_names.append(line.strip())

#     return dataset_files, dataset_file_names

def load_checkpoint(path: pathlib.Path):
    if not path.is_file():
        best_path = pathlib.Path(path, 'best-checkpoint.pth.tar')
        if best_path.is_file():
            path = best_path
        else:
            path = pathlib.Path(path, 'checkpoint.pth.tar')

    if not path.is_file():
        return
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=2,
        id2label=id2label, 
        label2id=label2id,
        reshape_last_stage=True,
        ignore_mismatched_sizes=True,
    )

    state_dict = torch.load(path)    
    model.load_state_dict(state_dict['model_state'])

    return model

## Model

torch.set_float32_matmul_precision('high')

## Warmup

def warmup(warmup_iters):
    dummy_input = torch.rand((1, 3, model_data['patch_size'], model_data['patch_size']))
    dummy_input = dummy_input.to('cuda')

    for i in range(1, warmup_iters):
        model(dummy_input)

## Inference

dataset_types = ['train', 'validation', 'test']

for model_data in models_data:
    print(model_data)
    print(f'[DATA]: Preparing data for {model_data["model_name"]}!')

    for type in dataset_types:
        print(f'[DATA]: Preparing model directory for {type}!')
        prepare_directory(model_data, type)

        print(f'[DATA]: Started preloading images and labels for {type}!')
        imgs, _ = preload_image_data(r'dataset', r'imgs', False, model_data['patch_size'], f'{type}_dataset.txt')

        print(f'[PREDICTION]: Started loading model for {type}!')
        model = load_checkpoint(pathlib.Path(model_data['model_path'])).cuda()
        model.eval()

        print('[PREDICTION]: Model loaded! Startin model warmup for 15 iterations.')
        warmup(15)

        print(f'[PREDICTION]: Model warmed up! Starting prediction on {len(imgs)} image(s).')
        pbar = tqdm.tqdm(enumerate(imgs), total=len(imgs))

        for i, img in pbar:
            # Save prediction
            data = val_transforms(image=img)
            data['image'] = data['image'].float().unsqueeze(0).cuda()

            with torch.no_grad():
                outputs =   model(data['image'])
                masks_pred = torch.nn.functional.interpolate(outputs.logits, size=(model_data['patch_size'], model_data['patch_size']), mode="bilinear", align_corners=False).squeeze(0).cpu()
                masks_pred = torch.sigmoid(masks_pred[1]).numpy() * 255.0

            path = get_image_directory(model_data, type).resolve()
            filename = pathlib.Path(path, f'{i}.png')
            cv2.imwrite(str(filename), masks_pred)

        pbar.close()
        print('[PREDICTION]: Finished prediction on provided images!\n')

