
"""
五组检测预测 Head
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x)


class DetectionHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=10):
        super().__init__()

        self.heatmap = PredictionHead(in_channels, num_classes)
        self.offset = PredictionHead(in_channels, 2)
        self.height = PredictionHead(in_channels, 1)
        self.dimensions = PredictionHead(in_channels, 3)
        self.rotation = PredictionHead(in_channels, 2)

        # Low initial probability for object centers
        nn.init.constant_(self.heatmap.net[-1].bias, -2.19)

    def forward(self, bev):
        return {
            "heatmap": self.heatmap(bev),
            "offset": self.offset(bev),
            "height": self.height(bev),
            "dimensions": self.dimensions(bev),
            "rotation": self.rotation(bev)
        }
