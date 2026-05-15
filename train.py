import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

from model import PetClassifier

BATCH_SIZE = 64
EPOCHS = 30
LR = 1.9e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# helper function for trimap cropping
def crop_with_trimap(image, trimap, padding=0.10):
    mask = np.array(trimap)

    # foreground + boundary
    pet_mask = mask != 2

    ys, xs = np.where(pet_mask)

    if len(xs) == 0 or len(ys) == 0:
        return image

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    w = x2 - x1
    h = y2 - y1

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.width, x2 + pad_x)
    y2 = min(image.height, y2 + pad_y)

    return image.crop((x1, y1, x2, y2))

# training transforms
train_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# validation transforms
val_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

base_dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    download=True
)

# deterministic validation split
indices = np.arange(len(base_dataset))

np.random.seed(42)
np.random.shuffle(indices)

split = int(0.9 * len(indices))

train_idx = indices[:split]
val_idx = indices[split:]

train_dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    transform=train_transform,
    download=False
)

val_dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    transform=val_transform,
    download=False
)

train_data = Subset(train_dataset, train_idx)
val_data = Subset(val_dataset, val_idx)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE
)

model = PetClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

best_val = 0

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    scheduler.step()

    train_acc = 100 * correct / total

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            preds = outputs.argmax(1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch {epoch+1}: "
        f"loss={total_loss:.2f}, "
        f"train_acc={train_acc:.2f}%, "
        f"val_acc={val_acc:.2f}%"
    )

    # save best validation model
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved new best model")

print("Training finished.")