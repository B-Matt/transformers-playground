import torch
import math

import numpy as np
import torchmetrics.functional as F
import segmentation_models_pytorch.utils.meter as meter

from tqdm import tqdm


@torch.inference_mode()
def validate(net, dataloader, device, gpu_id, epoch, wandb_log):
    net.eval()
    num_val_batches = len(dataloader)
    loss_meter = meter.AverageValueMeter()
    reports_data = {
        'Dice Score': [],
        'IoU Score': [],
    }

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', position=1, unit='batch', leave=False):
        # Get Batch Of Images
        batch_image = batch['image'].to(device, non_blocking=True)
        batch_mask = batch['mask'].to(device, non_blocking=True)
        threshold = 0.5

        with torch.no_grad():
            outputs = net(pixel_values=batch_image, labels=batch_mask)
            loss, logits = outputs.loss, outputs.logits
            loss_meter.add(loss.mean().item())

            if gpu_id == 0:
                mask_pred = torch.nn.functional.interpolate(logits, size=batch_mask.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
                dice_score = F.dice(mask_pred, batch_mask.long(), threshold=threshold, ignore_index=0).item()
                jaccard_index = F.classification.binary_jaccard_index(mask_pred, batch_mask.long(), threshold=threshold, ignore_index=0).item()

                if math.isnan (dice_score):
                    dice_score = 0.0

                if math.isnan (jaccard_index):
                    jaccard_index = 0.0

                reports_data['Dice Score'].append(dice_score)
                reports_data['IoU Score'].append(jaccard_index)

    if gpu_id == 0:
        # Update WANDB
        wandb_log.log({
            'Loss [validation]': loss_meter.mean,
            'IoU Score [validation]': np.mean(reports_data['IoU Score']),
            'Dice Score [validation]': np.mean(reports_data['Dice Score']),
        }, step=epoch)

    net.train()
    return loss_meter.mean
