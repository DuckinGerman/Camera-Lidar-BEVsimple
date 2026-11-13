"""
	Concatenate + Conv2D
"""

import torch
from torch import nn


class BEVFusion(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )

    def forward(self, camera_bev, lidar_bev):
        if camera_bev.shape != lidar_bev.shape:
            raise ValueError(f'BEV shape mismatch: {camera_bev.shape} vs {lidar_bev.shape}')
        return self.fusion(torch.cat([camera_bev, lidar_bev], dim=1))
