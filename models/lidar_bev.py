"""
点云体素化、Voxel Encoder、Sparse CNN
"""

import os
import numpy as np
import torch
from torch import nn

# Windows: PyTorch ships CUDA runtime DLLs required by spconv.
_DLL_HANDLE = None
if os.name == 'nt':
    _DLL_HANDLE = os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), 'lib'))
import spconv.pytorch as spconv


class VoxelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())

    def forward(self, points, inverse, count):
        f = self.mlp(points)
        out = torch.full((count, f.shape[1]), -torch.inf, device=f.device, dtype=f.dtype)
        return out.scatter_reduce(0, inverse[:, None].expand_as(f), f, reduce='amax', include_self=True)


class LiDARBEV(nn.Module):
    """Single-sample nuScenes point cloud [N,4] -> [1,64,80,80], indexed [Y,X]."""
    def __init__(self):
        super().__init__()
        self.voxel_encoder = VoxelEncoder()
        self.backbone = spconv.SparseSequential(
            spconv.SubMConv3d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(),
            spconv.SubMConv3d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.compression = nn.Sequential(nn.Conv2d(640, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.register_buffer('xyz_min', torch.tensor([-20., -20., -5.]))
        self.register_buffer('xyz_max', torch.tensor([20., 20., 3.]))
        self.register_buffer('voxel_size', torch.tensor([.5, .5, .4]))

    def forward(self, points):
        if not isinstance(points, torch.Tensor):
            points = torch.as_tensor(points, dtype=torch.float32, device=self.xyz_min.device)
        points = points.to(device=self.xyz_min.device, dtype=torch.float32)
        valid = ((points[:, :3] >= self.xyz_min) & (points[:, :3] < self.xyz_max)).all(1)
        points = points[valid]
        if len(points) == 0:
            raise ValueError('No LiDAR points inside configured voxel range')
        xyz = torch.floor((points[:, :3] - self.xyz_min) / self.voxel_size).int()
        zyx = xyz[:, [2, 1, 0]]
        coords, inverse = torch.unique(zyx, dim=0, return_inverse=True, sorted=True)
        f = self.voxel_encoder(points, inverse.long(), len(coords))
        batch = torch.zeros((len(coords), 1), dtype=torch.int32, device=points.device)
        indices = torch.cat([batch, coords], 1).int()
        sparse = spconv.SparseConvTensor(f, indices, [20, 80, 80], 1)
        dense = self.backbone(sparse).dense()  # [B,C,Z,Y,X]
        b, c, z, h, w = dense.shape
        return self.compression(dense.reshape(b, c*z, h, w))
