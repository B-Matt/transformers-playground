import os
import sys
import torch
import wandb
import pathlib
import datetime
import argparse
 
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from argparse import Namespace
 
import numpy as np
import albumentations as A
import torchmetrics.functional as F
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.meter as meter
 
from tqdm import tqdm
from pathlib import Path
from pipeline.validation import validate
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from utils.plots import plot_img_and_mask
 
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
 
from utils.dataset import Dataset, DatasetCacheType, DatasetType
from utils.early_stopping import YOLOEarlyStopping
 
# Logging
from utils.logging import logging
 
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
 
torch.multiprocessing.set_sharing_strategy('file_system')
 
class Trainer:
    def __init__(self, gpu_id, args, net, device, processor):
        assert net is not None
        self.model = net
        self.processor = processor
 
        self.gpu_id = gpu_id
        self.args = args
        self.start_epoch = 0
        self.check_best_cooldown = 0
        self.device = device
        self.project_name = 'maskformer'
 
        self.get_augmentations()
        self.get_loaders()
 
        self.optimizer = torch.optim.AdamW(self.model.parameters(), weight_decay=self.args.weight_decay, eps=self.args.adam_eps, lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        self.early_stopping = YOLOEarlyStopping(patience=30)
 
        self.metrics = [
            F.dice,
            F.jaccard_index,
        ]
        self.loss_meter = meter.AverageValueMeter()
        self.metrics_meters = { metric.__name__: meter.AverageValueMeter() for metric in self.metrics }
        self.metrics_logs = {
            'dice': 0.0,
            'jaccard_index': 0.0
        }
 
        if self.args.load_model:
            self.load_checkpoint(Path(self.args.load_model))
            self.model.to(self.device)
 
    def train_setup(self):
        if self.gpu_id != 0:
            return
 
        log.info(f'''[TRAINING]:
            Model:           {self.args.model}
            Encoder:         {self.args.encoder}
            Resolution:      {self.args.patch_size}x{self.args.patch_size}
            Epochs:          {self.args.epochs}
            Batch size:      {self.args.batch_size}
            Patch size:      {self.args.patch_size}
            Learning rate:   {self.args.lr}
            Training size:   {int(len(self.train_dataset))}
            Validation size: {int(len(self.val_dataset))}
            Checkpoints:     {self.args.save_checkpoints}
            Device:          {self.device.type}
            Mixed Precision: {self.args.use_amp}
            Using DDP:       {self.args.use_ddp}
        ''')
 
        wandb_log = wandb.init(project=self.project_name, entity='firebot031')
        wandb_log.config.update(
            dict(
                epochs=self.args.epochs,
                batch_size=self.args.batch_size,
                learning_rate=self.args.lr,
                save_checkpoint=self.args.save_checkpoints,
                patch_size=self.args.patch_size,
                amp=self.args.use_amp,
                weight_decay=self.args.weight_decay,
                adam_epsilon=self.args.adam_eps,
                encoder=self.args.encoder,
                model=self.args.model,
            )
        )
 
        self.run_name = wandb.run.name if wandb.run.name is not None else f'{self.args.model}-{self.args.encoder}-{self.args.batch_size}-{self.args.patch_size}'
        save_path = Path(f'checkpoints/{self.project_name}', self.run_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        return wandb_log
 
    def get_augmentations(self):
        self.train_transforms = A.Compose(
            [
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, p=0.5),
                #A.CoarseDropout(max_holes=6, max_height=12, max_width=12, min_holes=1, p=0.5),
                #A.ShiftScaleRotate(shift_limit=0.09, rotate_limit=0, p=0.2),
                A.OneOf(
                    [
                        A.GridDistortion(distort_limit=0.1, p=0.5),
                        A.OpticalDistortion(distort_limit=0.08, shift_limit=0.4, p=0.5),
                    ],
                    p=0.0
                ),
                #A.Perspective(scale=(0.02, 0.07), p=0.5),
 
                # Color transforms
                A.ColorJitter(
                    brightness=0, contrast=0, saturation=0.12, hue=0.01, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.05, 0.20), contrast_limit=(-0.05, 0.20), p=0.6
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 20.0), p=0.5),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.02, 0.09), p=0.5),
                    ],
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(5, 7), p=0.39),
                A.ToFloat(),
                ToTensorV2()
            ]
        )
 
        self.val_transforms = A.Compose(
            [
                A.ToFloat(),
                ToTensorV2(),
            ],
        )
 
    def get_loaders(self):
        self.train_dataset = Dataset(
            data_dir=r'dataset',
            img_dir=r'imgs',
            type=DatasetType.TRAIN, 
            patch_size=self.args.patch_size,
            transform=self.train_transforms,
            processor=self.processor
        )
        self.val_dataset = Dataset(
            data_dir=r'dataset',
            img_dir=r'imgs', 
            type=DatasetType.VALIDATION,
            patch_size=self.args.patch_size,
            transform=self.val_transforms,
            processor=self.processor
        )
 
        # Get Loaders    
        self.train_loader = DataLoader(
            self.train_dataset, 
            num_workers=self.args.workers,
            batch_size=self.args.batch_size,
            pin_memory=self.args.pin_memory,
            shuffle=True if not self.args.use_ddp else False, 
            drop_last=True if not self.args.use_ddp else False,
            sampler=DistributedSampler(self.train_dataset) if self.args.use_ddp else None,
            persistent_workers=self.args.workers > 0,
            collate_fn=collate_fn
        )
 
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            shuffle=False,
            drop_last=False,
            sampler=DistributedSampler(self.val_dataset) if self.args.use_ddp else None,
            persistent_workers=self.args.workers > 0,
            collate_fn=collate_fn
        )
 
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not self.args.save_checkpoints or self.gpu_id != 0:
            return
 
        model = self.model
        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
 
        state = {
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch
        }
 
        if is_best is False:
            # log.info('[SAVING MODEL]: Model checkpoint saved!')
            torch.save(state, Path(f'checkpoints/{self.project_name}', self.run_name, 'checkpoint.pth.tar'))
 
        if is_best:
            log.info('[SAVING MODEL]: Saving checkpoint of best model!')
            torch.save(state, Path(f'checkpoints/{self.project_name}', self.run_name, 'best-checkpoint.pth.tar'))
 
    def load_checkpoint(self, path: Path):
        log.info('[LOADING MODEL]: Started loading model checkpoint!')
 
        if not path.is_file():
            best_path = Path(path, 'best-checkpoint.pth.tar')
            if best_path.is_file():
                path = best_path
            else:
                path = Path(path, 'checkpoint.pth.tar')
 
        if not path.is_file():
            return
 
        state_dict = torch.load(path)
 
        if 'epoch' in state_dict and 'model_state' in state_dict and 'model_name' in state_dict:
            self.start_epoch = state_dict['epoch']
            self.model.load_state_dict(state_dict['model_state'])
            self.optimizer.load_state_dict(state_dict['optimizer_state'])
            self.optimizer.name = state_dict['optimizer_name']
            log.info(f"[LOADING MODEL]: Loaded model with stats: epoch ({state_dict['epoch']}), time ({state_dict['time']})!")
        else:
            self.start_epoch = 0
            self.model.load_state_dict(state_dict)
            log.info(f"[LOADING MODEL]: Loaded model!")        
 
    def train(self):
        wandb_log = self.train_setup()
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        criterion = torch.nn.CrossEntropyLoss() if self.args.classes > 1 else torch.nn.BCEWithLogitsLoss()
        criterion = criterion.to(device=self.device)
 
        global_step = 0
        last_best_score = float('inf')
        masks_pred = []
 
        torch.cuda.empty_cache()
        for epoch in range(self.start_epoch, self.args.epochs):
            val_loss = 0.0
            with tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.args.epochs}', unit='img', position=0, disable=self.gpu_id != 0) as progress_bar:
                for batch in progress_bar:
                    loss = 0.0
                    outputs = None
                    self.optimizer.zero_grad(set_to_none=True)
 
                    # Get Batch Of Images
                    batch_image = batch['original_images'][0].to(device, non_blocking=True)
                    batch_mask = batch['original_segmentation_maps'][0].to(device, non_blocking=True)
 
                    # Predict
                    with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                        outputs = self.model(
                            pixel_values=batch["pixel_values"].to(device),
                            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                            class_labels=[labels.to(device) for labels in batch["class_labels"]],
                        )
                        loss = outputs.loss
                        self.loss_meter.add(loss.item())
 
                    # Scale Gradients
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 255.0)
 
                    grad_scaler.step(self.optimizer)
                    grad_scaler.update()
                    self.scheduler.step()
 
                    # Show batch progress to terminal
                    global_step += 1
 
                    # Evaluation of training
                    eval_step = (int(len(self.train_dataset)) // (self.args.eval_step * self.args.batch_size))
                    if eval_step > 0 and global_step % eval_step == 0:
                        val_loss = validate(self.model, self.val_loader, self.device, self.gpu_id, epoch, wandb_log, self.processor, self.args)
 
                        if self.gpu_id == 0:
                            target_sizes = [(args.patch_size, args.patch_size)] * outputs.class_queries_logits.shape[0]
                            masks_pred = self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
 
                            jaccard_index = F.classification.binary_jaccard_index(masks_pred, batch_mask.long(), threshold=0.5, ignore_index=0).cpu().detach().numpy()
                            self.metrics_meters['jaccard_index'].add(jaccard_index)
 
                            dice = F.dice(masks_pred, batch_mask.long(), threshold=0.5, ignore_index=0).cpu().detach().numpy()
                            self.metrics_meters['dice'].add(dice)
                            self.metrics_logs = {k: v.mean for k, v in self.metrics_meters.items()}
 
                        if self.args.use_ddp:
                            torch.distributed.barrier()
 
                        self.early_stopping(epoch, val_loss)
                        if epoch >= self.check_best_cooldown and val_loss < last_best_score:
                            self.save_checkpoint(epoch, True)
                            last_best_score = val_loss
 
                        # Update WANDB with Images
                        if self.gpu_id == 0:
                            try:
                                wandb_log.log({
                                    'Images [training]': {
                                        'Image': wandb.Image(batch_image.cpu()),
                                        'Ground Truth': wandb.Image(batch_mask.squeeze(0).detach().cpu().numpy()),
                                        'Prediction': wandb.Image(masks_pred.detach().cpu().float().numpy()),
                                    },
                                }, step=epoch)
                            except Exception as e:
                                print('Wandb error: ', e)
 
                # Update WANDB
                if self.gpu_id == 0:
                    # Update Progress Bar
                    progress_bar.set_postfix(**{})
                    progress_bar.close()
                    wandb_log.log({
                        'Learning Rate': self.optimizer.param_groups[0]['lr'],
                        'Epoch': epoch,
                        'Loss [training]': self.loss_meter.mean,
                        'IoU Score [training]': self.metrics_logs['jaccard_index'],
                        'Dice Score [training]': self.metrics_logs['dice'],
                    }, step=epoch)
 
                    # Saving last model
                    if self.args.save_checkpoints:
                        self.save_checkpoint(epoch, False)
 
                    # Early Stopping
                    if self.early_stopping.early_stop:
                        self.save_checkpoint(epoch, False)
                        log.info(f'[TRAINING]: Early stopping training at epoch {epoch}!')
                        break
 
        # Push average training metrics
        if self.gpu_id == 0:
            wandb_log.finish()
 
            if self.args.use_ddp:
                torch.distributed.barrier()
                sys.exit(0)
 
def ddp_trainer_main(rank: int, world_size: int, args: Namespace, net: any, processor: any):
    torch.cuda.set_device(rank)
    init_process_group('nccl', rank=rank, world_size=world_size)
 
    net = net.to(rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)
 
    try:
        trainer = Trainer(rank, args, net, torch.device(f'cuda:{rank}'), processor)
        trainer.train()
    except KeyboardInterrupt:
        try:
            destroy_process_group()
            sys.exit(0)
        except KeyboardInterrupt: 
            os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')")
 
    destroy_process_group()
 
def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
 
    batch = processor(
        images,
        segmentation_maps=segmentation_maps,
        task_inputs=["semantic"],
        return_tensors="pt",
    )
 
    batch["original_images"] = inputs[0]
    batch["original_segmentation_maps"] = inputs[1]
    return batch
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MaskFormer', help='Which model you want to train?')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--adam-eps', nargs='+', type=float, default=1e-3, help='Adam epsilon')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay that is used for AdamW')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--encoder', default="", help='Backbone encoder')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of DataLoader workers')
    parser.add_argument('--classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--patch-size', type=int, default=800, help='Patch size')
    parser.add_argument('--pin-memory', type=bool, default=True, help='Use pin memory for DataLoader?')
    parser.add_argument('--eval-step', type=int, default=1, help='Run evaluation every # step')
    parser.add_argument('--load-model', default="", help='Load model from directories')
    parser.add_argument('--save-checkpoints', action='store_true', help='Save checkpoints after every epoch?')
    parser.add_argument('--use-amp', action='store_true', help='Use Pytorch Automatic Mixed Precision?')
    parser.add_argument('--use-dp', action='store_true', help='Use Pytorch Data Parallel?')
    parser.add_argument('--use-ddp', action='store_true', help='Use Pytorch Distributed Data Parallel?')
    parser.add_argument('--search-files', type=bool, default=False, help='Should DataLoader search your files for images?')
    parser.add_argument('--dataset', default='fire', help='Should DataLoader search your files for images?')
    args = parser.parse_args()
 
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:512'
 
    id2label = { 0: 'background', 1: 'fire' }
    label2id = { 'background': 0, 'fire': 1 }
 
    model_name = "facebook/maskformer-swin-base-ade" #"facebook/mask2former-swin-base-ade-semantic"
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        do_reduce_labels=False,
        ignore_index=0
    )
    net = AutoModelForUniversalSegmentation.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
 
    processor.num_text = 0 #net.config.num_queries - net.config.text_encoder_n_ctx
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
 
    if args.use_ddp:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'
 
        world_size = torch.cuda.device_count()
        mp.spawn(ddp_trainer_main, args=(world_size, args, net, processor), nprocs=world_size)
 
    if not args.use_ddp:
        net = net.to(device)
        if args.use_dp:
            net = torch.nn.DataParallel(net)
 
        trainer = Trainer(0, args, net, device, processor)
 
        try:
            trainer.train()
        except KeyboardInterrupt:
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
 
    # End Training
    logging.info('[TRAINING]: Training finished!')
    torch.cuda.empty_cache()
 
    net = None
    trainer = None
 
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
