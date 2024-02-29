"""
    Data preparation for the scientific article.
"""
import os
import math
import tqdm
import torch
import pathlib

import numpy as np
import torchmetrics.functional as F

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

def calc_metrics(pred, target, threshold, resolution):
    dice_score = F.dice(pred, target, threshold=threshold, ignore_index=0).item()
    jaccard_index = F.classification.binary_jaccard_index(pred, target, threshold=threshold, ignore_index=0).item()
    conf_matrix = F.classification.binary_confusion_matrix(pred, target, threshold=threshold, ignore_index=0, normalize='none')
    conf_matrix = np.array(conf_matrix.tolist())

    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tp = conf_matrix[1][1]

    if math.isnan (dice_score):
        dice_score = 0.0

    if math.isnan (jaccard_index):
        jaccard_index = 0.0

    total_error = (fn + fp) / (resolution * resolution)
    return dice_score, jaccard_index, total_error

# Main
model_data_types = ['validation']
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

for model_data in models_data:
    for type in model_data_types:
        imgs, _  = preload_image_data(r'dataset', r'imgs', False, model_data['patch_size'], f'{type}_dataset.txt')
        masks, _ = preload_image_data(r'dataset', r'imgs', True, model_data['patch_size'], f'{type}_dataset.txt')

        model = load_model(model_data['model_path']).to('cuda:0')
        model.eval()
        warmup(model, 10, model_data['patch_size'])

        timings = []
        pbar = tqdm.tqdm(enumerate(imgs), total=len(imgs))

        reports_data = {
            'Dice Score': {
                0.2: [],
                0.3: [],
                0.4: [],
                0.5: [],
                0.6: []
            },
            'IoU Score': {
                0.2: [],
                0.3: [],
                0.4: [],
                0.5: [],
                0.6: []
            },
            'Total Error': {
                0.2: [],
                0.3: [],
                0.4: [],
                0.5: [],
                0.6: []
            },
        }

        for i, img in pbar:
            patch_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to('cuda:0')            

            with torch.no_grad():
                outputs = model(pixel_values=patch_tensor)
                mask_pred = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=(model_data['patch_size'], model_data['patch_size']),
                    mode="bilinear",
                    align_corners=False
                ).argmax(dim=1).squeeze(0).cpu()

                for threshold in thresholds:
                    target_mask = masks[i] / 255.0
                    target_mask = target_mask.astype(np.uint8)
                    dice_score, jaccard_index, total_error = calc_metrics(mask_pred, torch.from_numpy(target_mask), threshold, model_data['patch_size'])

                    reports_data[f'Dice Score'][threshold].append(dice_score)
                    reports_data[f'IoU Score'][threshold].append(jaccard_index)
                    reports_data[f'Total Error'][threshold].append(total_error)

        pbar.close()

        print('---------------------')
        for threshold in thresholds:
            print(
               type,
               model_data['model_name'],
               threshold * 100,
               round(np.mean(reports_data[f'Dice Score'][threshold]), 3),
               round(np.mean(reports_data[f'IoU Score'][threshold]), 3),
               round(np.mean(reports_data[f'Total Error'][threshold]), 5)
            )
        print('---------------------')
