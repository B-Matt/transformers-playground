import math
import torch
import wandb

import numpy as np
import torchmetrics.functional as F
import segmentation_models_pytorch.utils.meter as meter

from tqdm import tqdm

from utils.plots import plot_img_and_mask
 
@torch.inference_mode()
def validate(net, dataloader, device, gpu_id, epoch, wandb_log, processor, args):
    net.eval()
 
    num_val_batches = len(dataloader)
    loss_meter = meter.AverageValueMeter()
    reports_data = {
        'Dice Score': [],
        'IoU Score': [],
    }
 
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation', position=1, unit='batch', leave=False):
        threshold = 0.5

        with torch.no_grad():
            outputs = net(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            print(batch['pixel_values'].shape, batch['mask_labels'][0].shape, batch['class_labels'][0].shape)

            loss = outputs.loss
            loss_meter.add(loss.cpu())
 
            if gpu_id == 0:
                target_sizes = [(args.patch_size, args.patch_size)] * outputs.class_queries_logits.shape[0]
                mask_pred = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

                for i in range(len(batch['mask_labels'])):
                    #plot_img_and_mask(batch['original_images'][i].cpu().squeeze(0).permute(1, 2, 0).numpy(), mask_pred[i].detach().cpu().float().numpy())
                    
                    batch_image = batch['pixel_values'][i].to(device)
                    batch_mask = batch['mask_labels'][i].to(device)

                    print(mask_pred[i].unsqueeze(0).shape, batch_mask.shape)
    
                    dice_score = F.dice(
                        mask_pred[i],
                        batch_mask.long(),
                        threshold=threshold,
                        ignore_index=0
                    ).item()

                    jaccard_index = F.classification.binary_jaccard_index(
                        mask_pred[i],
                        batch_mask.long(),
                        threshold=threshold,
                        ignore_index=0
                    ).item()
    
                    if math.isnan (dice_score):
                        dice_score = 0.0
    
                    if math.isnan (jaccard_index):
                        jaccard_index = 0.0
    
                    reports_data['Dice Score'].append(dice_score)
                    reports_data['IoU Score'].append(jaccard_index)

    if gpu_id == 0:
        # Update WANDB
        try:
            wandb_log.log({
                'Loss [validation]': loss_meter.mean,
                'IoU Score [validation]': np.mean(reports_data['IoU Score']),
                'Dice Score [validation]': np.mean(reports_data['Dice Score']),
                'Images [training]': {
                    'Image': wandb.Image(batch_image.cpu()),
                    'Ground Truth': wandb.Image(batch_mask.squeeze(0).detach().cpu().numpy()),
                    'Prediction': wandb.Image(mask_pred.unsqueeze(0).detach().cpu().float().numpy() * 255.0),
                },
            }, step=epoch)
        except Exception as e:
            print('Wandb error: ', e)
 
    net.train()
    return loss_meter.mean
