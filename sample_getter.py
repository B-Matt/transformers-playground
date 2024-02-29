"""
    Data preparation for the scientific article.
"""
import os
import time
import tqdm
import torch
import pathlib

import numpy as np

from transformers import SegformerForSemanticSegmentation
from utils.prediction.evaluations import preload_image_data
from utils.prediction.predict import Prediction

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Vars
models_data = [
    {
        'model_name': 'SegFormer 256x256',
        'model_path': r'checkpoints/segformer/scarlet-totem-58/best-checkpoint.pth.tar',
        'patch_size': 256,
    },
    #{
    #    'model_name': 'SegFormer 640x640',
    #    'model_path': r'checkpoints/segformer/sandy-resonance-59/best-checkpoint.pth.tar',
    #    'patch_size': 640,
    #},
    #{
    #    'model_name': 'SegFormer 800x800',
    #    'model_path': r'checkpoints/segformer/morning-moon-60/best-checkpoint.pth.tar',
    #    'patch_size': 800,
    #},
]

# Functions
def prepare_directory(model_data, dataset_type):
    preped_data = pathlib.Path('playground', 'preped_data')
    model_output = pathlib.Path(preped_data, model_data['model_name'], dataset_type)

    # Create directory if it doesn't exists
    if not os.path.isdir(model_output):
        os.makedirs(model_output)

def get_image_directory(model_data, dataset_type):
    preped_data = pathlib.Path('playground', 'preped_data')
    model_output = pathlib.Path(preped_data, model_data['model_name'], dataset_type)
    return model_output

def load_model(model_path):
    id2label = { 0: 'background', 1: 'fire' }
    label2id = { 'background': 0, 'fire': 1 }

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3",
        num_labels=2,
        id2label=id2label, 
        label2id=label2id,
        reshape_last_stage=True,
        ignore_mismatched_sizes=True,
    )

    state_dict = torch.load(model_path, map_location=torch.device('cuda:0'))
    model.load_state_dict(state_dict['model_state'])
    return model

def warmup(model, warmup_iters, image_resolution):
    dummy_input = torch.rand((1, 3, image_resolution, image_resolution)).to('cuda')

    for i in range(1, warmup_iters):
        output = model(dummy_input)

# Main
model_data_types = ['validation']

for model_data in models_data:
    for type in model_data_types:
        imgs, _ = preload_image_data(r'dataset', r'imgs', False, model_data['patch_size'], f'{type}_dataset.txt')

        model = load_model(model_data['model_path']).to('cuda:0')
        model.eval()
        warmup(model, 10, model_data['patch_size'])

        timings = []
        pbar = tqdm.tqdm(enumerate(imgs), total=len(imgs))

        for i, img in pbar:
            patch_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to('cuda:0')

            timer_start = time.perf_counter()
            with torch.no_grad():
                outputs = model(pixel_values=patch_tensor)
                mask_pred = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=img.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                ).argmax(dim=1)

                torch.cuda.synchronize()
                timings.append(time.perf_counter() - timer_start)

        pbar.close()
        log.info(f'[PREDICTION - {model_data["model_name"]}]: Mean time: {(np.mean(timings) * 1000):.3f}ms | Standard deviation: {(np.std(timings) * 1000):.3f}ms')
