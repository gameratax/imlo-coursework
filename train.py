import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

from model import PetClassifier

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# training transforms
train_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
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

dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    download=True
)

# validation split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = random_split(
    dataset,
    [train_size, val_size]
)

train_data.dataset.transform = train_transform
val_data.dataset.transform = val_transform

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

# Adam optimizer with weight decay
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# gradually reduce learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
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