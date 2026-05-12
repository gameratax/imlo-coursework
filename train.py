import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

from model import PetClassifier

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# image preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# load official training split
train_data = OxfordIIITPet(
    root="./data",
    split="trainval",
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = PetClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

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

    train_acc = 100 * correct / total

    print(
        f"Epoch {epoch+1}: "
        f"loss={total_loss:.2f}, "
        f"train_acc={train_acc:.2f}%"
    )

# save trained model
torch.save(model.state_dict(), "best_model.pth")

print("Training finished.")