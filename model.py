import torch
import torch.nn as nn


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class PetClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # convolution blocks
        self.block1 = conv_block(3, 64)
        self.block2 = conv_block(64, 128)
        self.block3 = conv_block(128, 256)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        # classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            # reduce overfitting
            nn.Dropout(0.3),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x