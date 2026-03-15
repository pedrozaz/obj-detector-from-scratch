import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Input: 3 x 488 x 488 -> 64 x 224 x 224
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 64 x 112 x 112
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bottleneck (3 / 4) -> 256 x 28 x 28
            ConvBlock(192, 128, kernel_size=1, stride=1, padding=0),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bottleneck (5 / 6) - 512 x 14 x 14
            ConvBlock(256, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bottleneck (7 / 8) -> 1024 x 7 x 7
            ConvBlock(512, 512, kernel_size=1, stride=1, padding=0),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)

class DetectionHead(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes

        self.conv = nn.Sequential(
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)

class TinyDetector(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.backbone = Backbone()
        self.head = DetectionHead(grid_size, num_boxes, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions