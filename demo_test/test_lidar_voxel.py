
import os
import numpy as np
import torch

# Fix Windows CUDA DLL loading
torch_lib = os.path.join(
    os.path.dirname(torch.__file__), "lib"
)
dll_handle = os.add_dll_directory(torch_lib)

import spconv.pytorch as spconv

# -------------------------
# 1. Load nuScenes LiDAR
# -------------------------
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=r"D:\bevtrain\data\nuscenes",
    verbose=False
)

sample = nusc.sample[0]
lidar_token = sample["data"]["LIDAR_TOP"]
sd = nusc.get("sample_data", lidar_token)

lidar_path = os.path.join(nusc.dataroot, sd["filename"])

# nuScenes LiDAR: x, y, z, intensity, ring
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]

print("Raw points:", points.shape)

# -------------------------
# 2. Voxelization
# -------------------------
xyz_min = np.array([-20.0, -20.0, -5.0], dtype=np.float32)
xyz_max = np.array([20.0, 20.0, 3.0], dtype=np.float32)
voxel_size = np.array([0.5, 0.5, 0.4], dtype=np.float32)

mask = np.all(
    (points[:, :3] >= xyz_min) &
    (points[:, :3] < xyz_max),
    axis=1
)
points = points[mask]

coords_xyz = np.floor(
    (points[:, :3] - xyz_min) / voxel_size
).astype(np.int32)

# spconv coordinates: [batch, z, y, x]
coords_zyx = coords_xyz[:, [2, 1, 0]]

unique_coords, inverse = np.unique(
    coords_zyx, axis=0, return_inverse=True
)

# Aggregate mean point features per voxel
num_voxels = len(unique_coords)
# voxel_features = np.zeros((num_voxels, 4), dtype=np.float32)
# counts = np.bincount(inverse, minlength=num_voxels)

# np.add.at(voxel_features, inverse, points)
# voxel_features /= counts[:, None]

# print("Points in range:", len(points))
# print("Occupied voxels:", num_voxels)


import torch.nn as nn

# -------------------------
# Learnable Voxel Encoder
# -------------------------
class VoxelEncoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=16):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels),
            nn.ReLU()
        )

    def forward(self, points, inverse, num_voxels):
        # points: [N, 4]
        # inverse: [N], voxel index for each point

        point_features = self.mlp(points)

        voxel_features = torch.full(
            (num_voxels, point_features.shape[1]),
            -torch.inf,
            device=points.device
        )

        indices = inverse[:, None].expand_as(point_features)

        voxel_features = voxel_features.scatter_reduce(
            0, indices, point_features,
            reduce="amax",
            include_self=True
        )

        return voxel_features


device = "cuda"

encoder = VoxelEncoder().to(device)

points_tensor = torch.from_numpy(points).float().to(device)
inverse_tensor = torch.from_numpy(inverse).long().to(device)

voxel_features = encoder(
    points_tensor,
    inverse_tensor,
    num_voxels
)

print("Learnable voxel features:", voxel_features.shape)


# -------------------------
# 3. SparseConvTensor
# -------------------------
device = "cuda"

batch_idx = np.zeros((num_voxels, 1), dtype=np.int32)
indices = np.concatenate([batch_idx, unique_coords], axis=1)

x = spconv.SparseConvTensor(
    features=voxel_features,
    indices=torch.from_numpy(indices).to(device),
    spatial_shape=[20, 80, 80],
    batch_size=1
)

print("Sparse features:", tuple(x.features.shape))
print("Sparse indices:", tuple(x.indices.shape))
print("Spatial shape:", x.spatial_shape)
print("SparseConvTensor created OK")



# -------------------------
# 4. Sparse 3D CNN
# -------------------------
import torch.nn as nn

model = spconv.SparseSequential(
    spconv.SubMConv3d(
        16, 16, kernel_size=3,
        padding=1, bias=False
    ),
    nn.BatchNorm1d(16),
    nn.ReLU(),

    spconv.SubMConv3d(
        16, 32, kernel_size=3,
        padding=1, bias=False
    ),
    nn.BatchNorm1d(32),
    nn.ReLU()
).to(device).eval()

# with torch.no_grad():
    # output = model(x)

    # Sparse 3D -> Dense 3D
    # dense = output.dense()

    # [B, C, Z, Y, X] -> [B, C*Z, Y, X]
    # B, C, Z, H, W = dense.shape
    # bev = dense.reshape(B, C * Z, H, W)

bev_compression = nn.Sequential(
    nn.Conv2d(640, 64, kernel_size=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
).to(device)


model.train()
encoder.train()
bev_compression.train()

output = model(x)
dense = output.dense()

B, C, Z, H, W = dense.shape
bev = dense.reshape(B, C * Z, H, W)

lidar_bev = bev_compression(bev)


bev_compression = nn.Sequential(
    nn.Conv2d(640, 64, kernel_size=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
).to(device)

lidar_bev = bev_compression(bev)

print("Final LiDAR BEV:", tuple(lidar_bev.shape))





# -------------------------
# Backward propagation test
# -------------------------

loss = lidar_bev.square().mean()

loss.backward()

print("Loss:", loss.item())

print(
    "Voxel Encoder gradient:",
    encoder.mlp[0].weight.grad.norm().item()
)

print(
    "Sparse CNN gradient:",
    model[0].weight.grad.norm().item()
)

print(
    "BEV Compression gradient:",
    bev_compression[0].weight.grad.norm().item()
)

print(
    "Peak GPU memory (MB):",
    round(torch.cuda.max_memory_allocated() / 1024**2, 1)
)

print("LiDAR backward OK")
