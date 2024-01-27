import os
import torch
import wandb
import pathlib
import evaluate
import utils.dataset

import numpy as np
import albumentations as A
import torchmetrics.functional as F
import segmentation_models_pytorch.utils.meter as meter

from albumentations.pytorch import ToTensorV2
from transformers import TrainingArguments, Trainer
from utils.early_stopping import YOLOEarlyStopping


# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class CustomTrainer:
    def __init__(self, net, image_processor, args):
        assert net is not None
        self.model = net
        self.image_processor = image_processor
        self.args = args

        self.start_epoch = 0
        self.check_best_cooldown = 0
        self.project_name = "segformer"

        self.get_augmentations()
        self.get_loaders()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=self.args.weight_decay,
            eps=self.args.adam_eps,
            lr=self.args.lr,
        )
        self.early_stopping = YOLOEarlyStopping(patience=30)
        self.metric = evaluate.load("mean_iou")

    def get_augmentations(self):
        self.train_transforms = A.Compose(
            [
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, p=0.5),
                A.CoarseDropout(
                    max_holes=6, max_height=12, max_width=12, min_holes=1, p=0.5
                ),
                A.ShiftScaleRotate(shift_limit=0.09, rotate_limit=0, p=0.2),
                A.OneOf(
                    [
                        A.GridDistortion(distort_limit=0.1, p=0.5),
                        A.OpticalDistortion(distort_limit=0.08, shift_limit=0.4, p=0.5),
                    ],
                    p=0.6,
                ),
                A.Perspective(scale=(0.02, 0.07), p=0.5),
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
                        A.ISONoise(
                            color_shift=(0.01, 0.05), intensity=(0.02, 0.09), p=0.5
                        ),
                    ],
                    p=0.5,
                ),
                A.GaussianBlur(blur_limit=(5, 7), p=0.39),
                ToTensorV2(),
            ]
        )

        self.val_transforms = A.Compose(
            [
                ToTensorV2(),
            ],
        )

    def get_loaders(self):
        self.train_dataset = utils.dataset.Dataset(
            data_dir=r"dataset",
            img_dir=r"imgs",
            type=utils.dataset.DatasetType.TRAIN,
            patch_size=self.args.patch_size,
            transform=self.train_transforms,
            image_processor=self.image_processor,
        )
        self.val_dataset = utils.dataset.Dataset(
            data_dir=r"dataset",
            img_dir=r"imgs",
            type=utils.dataset.DatasetType.VALIDATION,
            patch_size=self.args.patch_size,
            transform=self.val_transforms,
            image_processor=self.image_processor,
        )

    def compute_metrics(self, eval_pred):
        dice_meter = meter.AverageValueMeter()
        iou_meter = meter.AverageValueMeter()

        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)

            logits_tensor = torch.nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
            target = torch.from_numpy(labels / 255).long()

            for i in range(len(logits_tensor)):
                dice_meter.add(
                    F.dice(
                        logits_tensor[i],
                        target[i],
                        num_classes=self.args.classes,
                        ignore_index=0
                    )
                )

                iou_meter.add(
                    F.jaccard_index(
                        logits_tensor[i],
                        target[i],
                        task="binary",
                        num_classes=self.args.classes,
                        ignore_index=0
                    )
                )

        self.wandb_log.log({
            "Ground Truth": wandb.Image(labels[0]),
            "Prediction": wandb.Image(logits_tensor[0].detach().cpu().numpy()),
        })

        return {
            'iou': iou_meter.mean,
            'dice': dice_meter.mean,
        }

    def train_setup(self):
        log.info(
            f"""[TRAINING]:
            Model:           SegFormer
            Resolution:      {self.args.patch_size}x{self.args.patch_size}
            Epochs:          {self.args.epochs}
            Batch size:      {self.args.batch_size}
            Patch size:      {self.args.patch_size}
            Learning rate:   {self.args.lr}
            Training size:   {int(len(self.train_dataset))}
            Validation size: {int(len(self.val_dataset))}
        """
        )

        self.wandb_log = wandb.init(project=self.project_name, entity="firebot031")
        self.wandb_log.config.update(
            dict(
                epochs=self.args.epochs,
                batch_size=self.args.batch_size,
                learning_rate=self.args.lr,
                patch_size=self.args.patch_size,
                weight_decay=self.args.weight_decay,
            )
        )

        self.run_name = (
            wandb.run.name
            if wandb.run.name is not None
            else f"{self.args.model}-{self.args.encoder}-{self.args.batch_size}-{self.args.patch_size}"
        )
        save_path = pathlib.Path(f"checkpoints/{self.project_name}", self.run_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        train_args = TrainingArguments(
            output_dir=f"./checkpoints/{self.project_name}/{self.run_name}",
            overwrite_output_dir = True,
            ddp_backend="nccl" if self.args.use_ddp else None,
            learning_rate=self.args.lr,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            push_to_hub=False,
            dataloader_num_workers=self.args.workers,
            dataloader_persistent_workers=True,
            optim="adamw_torch",
            weight_decay=self.args.weight_decay,
            adam_epsilon=self.args.adam_eps,

            # Wandb
            report_to="wandb",
            run_name=self.run_name,
            logging_steps=1,
        )

        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            #tokenizer=self.image_processor,
            compute_metrics=self.compute_metrics,
            #data_collator=self.collate_fn,
        )

        trainer.train()
        wandb.finish()
