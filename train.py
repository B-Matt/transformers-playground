import torch
import torchvision

import torch.optim as optim

from torch import nn
from models.vit import ViT
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop

from transformers import Trainer, TrainingArguments


# Hyperparameters
epochs = 10
base_lr = 10e-5
weight_decay = 0.03
batch_size = 4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Data Prep
train_transforms = Compose([RandomCrop(32, padding=4), Resize((224)), RandomHorizontalFlip(), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
train_data = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', train=True, download = True, transform = train_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
  # Setup model, optimizer, criterion & scheduler
  model = ViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
  ).to(device)
  optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
  criterion = nn.CrossEntropyLoss().to(device)
  scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=5)

  if torch.cuda.device_count() > 1:
    print(f'We will use multiple GPUs ({torch.cuda.device_count()})')
    model = nn.DataParallel(model).to(device)

  model.train()
  for epoch in tqdm(range(epochs), total=epochs, desc='Epochs'):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 2000 == 1999:
        print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
        running_loss = 0.0