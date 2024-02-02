import os
import torch
from transformers.trainer_callback import TrainerControl, TrainerState
import wandb
import pathlib
import argparse
import evaluate
import utils.dataset

import numpy as np
import albumentations as A

from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, jaccard_score

from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation, EarlyStoppingCallback

# Logging
from utils.logging import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

metric = evaluate.load("mean_iou")

run_name = ""

# Functions
def prepare_data(processor):
    # Augumentation
    train_transforms = A.Compose(
        [
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.5),
            A.OneOf(
                [
                    A.GridDistortion(distort_limit=0.1, p=0.5),
                    A.OpticalDistortion(distort_limit=0.08, shift_limit=0.4, p=0.5),
                ],
                p=0.0
            ),

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

    val_transforms = A.Compose(
        [
            A.ToFloat(),
            ToTensorV2(),
        ],
    )
    
    # Datasets
    train_dataset = utils.dataset.Dataset(
        data_dir=r"dataset",
        img_dir=r"imgs",
        type=utils.dataset.DatasetType.TRAIN,
        patch_size=args.patch_size,
        transform=train_transforms,
        processor=processor,
    )
    val_dataset = utils.dataset.Dataset(
        data_dir=r"dataset",
        img_dir=r"imgs",
        type=utils.dataset.DatasetType.VALIDATION,
        patch_size=args.patch_size,
        transform=val_transforms,
        processor=processor,
    )
    return train_dataset, val_dataset

def collate_fn(examples):
    # Get the pixel values, pixel mask, mask labels, and class labels
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_mask = torch.stack([example["pixel_mask"] for example in examples])
    mask_labels = [example["mask_labels"] for example in examples]
    class_labels = [example["class_labels"] for example in examples]

    #print(mask_labels, class_labels)
    #print(pixel_values.shape, pixel_mask.shape, len(mask_labels), len(class_labels))

    # Return a dictionary of all the collated features
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(type(eval_pred), eval_pred.__dict__.keys())
    print('aaaaa', len(predictions), len(labels))

    print(metric.compute(predictions=predictions, references=labels))
    return {
        'accuracy': accuracy_score(predictions, labels),
        'jaccard': jaccard_score(predictions, labels, average="weighted"),
    }

def train(model, train_dataset, val_dataset, args, processor):    
    train_args = TrainingArguments(
        output_dir=f"./checkpoints/{args.name.lower()}/{args.patch_size}x{args.patch_size}-{args.batch_size}",
        overwrite_output_dir = True,
        load_best_model_at_end=True,
        
        # DDP
        ddp_backend="nccl",
        ddp_find_unused_parameters=True,
        fp16=True,

        # Regular Hyperparameters
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False,
        dataloader_num_workers=args.workers,
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_eps,

        # Early Stopping
        metric_for_best_model="eval_loss",
        greater_is_better=False,            # We want to minimize validation loss

        # Pytorch v2
        #torch_compile=True,

        # Wandb
        report_to="wandb",
        run_name=run_name,
        logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,

        # Callbacks
        callbacks=[EarlyStoppingCallback(early_stopping_patience=30)]
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='Mask2Former', help='Training name')
    parser.add_argument('--model', type=str, default='facebook/mask2former-swin-tiny-cityscapes-semantic', help='Which model you want to train?')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--adam-eps', nargs='+', type=float, default=1e-3, help='Adam epsilon')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay that is used for AdamW')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--patch-size', type=int, default=800, help='Patch size')
    parser.add_argument('--workers', type=int, default=6, help='Dataloader workers')
    parser.add_argument('--use-ddp', action='store_true', help='Use Pytorch Distributed Data Parallel?')
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = args.name.lower()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:512'   

    id2label = { 1: 'fire' }
    label2id = { 'fire': 1 }

    processor = AutoImageProcessor.from_pretrained(
        args.model,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        do_reduce_labels=False,
        ignore_index=0,
        padding=False,
    )
    model = AutoModelForUniversalSegmentation.from_pretrained(
        args.model,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model = model.cuda()

    # Prepare data
    train_dataset, val_dataset = prepare_data(processor)

    # Logs
    if int(os.environ.get("LOCAL_RANK", -1)) == 0:
        log.info(
            f"""[TRAINING]:
                Model:           {args.name}
                Resolution:      {args.patch_size}x{args.patch_size}
                Epochs:          {args.epochs}
                Batch size:      {args.batch_size}
                Patch size:      {args.patch_size}
                Learning rate:   {args.lr}
                Training size:   {int(len(train_dataset))}
                Validation size: {int(len(val_dataset))}
            """
        )

        save_path = pathlib.Path(f"checkpoints/{args.name.lower()}", f"{args.patch_size}x{args.patch_size}-{args.batch_size}")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

    # Train
    train(model, train_dataset, val_dataset, args, processor)
