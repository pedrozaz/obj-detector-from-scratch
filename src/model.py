from pdb import Restart

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DetectionHead(nn.Module):
    def __init__(self, in_channels, grid_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * self.S * self.S, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, self.S * self.S * (self.B * 5 + self.C))
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)

class TinyDetector(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.backbone = nn.Sequential(*(list(resnet.children())[:-2]))

        self.downsample = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
        )

        self.head = DetectionHead(
            in_channels=512,
            grid_size=grid_size,
            num_boxes=num_boxes,
            num_classes=num_classes
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.downsample(features)
        predictions = self.head(features)
        return predictions