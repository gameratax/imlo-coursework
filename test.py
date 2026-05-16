import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import numpy as np
from PIL import Image
from model import PetClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def crop_with_trimap(image, trimap, padding=0.20):
    mask = np.array(trimap)

    # only definite pet pixels
    pet_mask = mask == 1

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


class PetTrimapDataset(Dataset):
    def __init__(self, split, transform=None, download=True):
        self.data = OxfordIIITPet(
            root="./data",
            split=split,
            target_types="category",
            download=download
        )

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        img_path = self.data._images[idx]
        filename = os.path.basename(img_path)

        trimap_path = os.path.join(
            "./data",
            "oxford-iiit-pet",
            "annotations",
            "trimaps",
            filename.replace(".jpg", ".png")
        )

        if os.path.exists(trimap_path):
            trimap = Image.open(trimap_path)
            image = crop_with_trimap(image, trimap)

        if self.transform:
            image = self.transform(image)

        # extra trimap mask channel
        if os.path.exists(trimap_path):
            trimap = Image.open(trimap_path).resize((160, 160))

            mask = np.array(trimap)

            # only definite pet pixels
            mask = (mask == 1).astype(np.float32)

            mask = torch.tensor(mask).unsqueeze(0)

        else:
            mask = torch.zeros((1, 160, 160), dtype=torch.float32)

        # RGB + mask channel
        image = torch.cat([image, mask], dim=0)

        return image, label


transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

trainval = PetTrimapDataset(
    split="trainval",
    transform=transform,
    download=True
)

test_data = PetTrimapDataset(
    split="test",
    transform=transform,
    download=True
)

train_loader = DataLoader(
    trainval,
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