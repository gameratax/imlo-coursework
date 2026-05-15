import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from model import PetClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_data = OxfordIIITPet(
    root="./data",
    split="trainval",
    transform=transform,
    download=True
)

test_data = OxfordIIITPet(
    root="./data",
    split="test",
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_data,
    batch_size=64
)

test_loader = DataLoader(
    test_data,
    batch_size=128
)

model = PetClassifier().to(device)

model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)

model.eval()


def evaluate(loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


train_acc = evaluate(train_loader)
test_acc = evaluate(test_loader)

print("\n==============================")
print(f"Train accuracy: {train_acc:.2f}%")
print(f"Test accuracy : {test_acc:.2f}%")
print("==============================")