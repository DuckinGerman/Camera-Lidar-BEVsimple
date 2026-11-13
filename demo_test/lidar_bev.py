import os
import numpy as np
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

DATAROOT = r"D:\bevtrain\data\nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATAROOT,
    verbose=False
)

sample = nusc.sample[0]
lidar_token = sample["data"]["LIDAR_TOP"]
lidar_path = nusc.get_sample_data_path(lidar_token)

pc = LidarPointCloud.from_file(lidar_path)

# LiDAR points: (N, 4)
points = pc.points.T

# BEV range and resolution
XY_MIN = -20.0
XY_MAX = 20.0
RES = 0.5
SIZE = int((XY_MAX - XY_MIN) / RES)

# Keep points inside BEV range
mask = (
    (points[:, 0] >= XY_MIN)
    & (points[:, 0] < XY_MAX)
    & (points[:, 1] >= XY_MIN)
    & (points[:, 1] < XY_MAX)
)

points = points[mask]

# World coordinates -> grid indices
ix = np.floor((points[:, 0] - XY_MIN) / RES).astype(int)
iy = np.floor((points[:, 1] - XY_MIN) / RES).astype(int)

height = np.full((SIZE, SIZE), -np.inf, dtype=np.float32)
density = np.zeros((SIZE, SIZE), dtype=np.float32)
intensity_sum = np.zeros((SIZE, SIZE), dtype=np.float32)

# Aggregate features for each cell
np.maximum.at(height, (ix, iy), points[:, 2])
np.add.at(density, (ix, iy), 1)
np.add.at(intensity_sum, (ix, iy), points[:, 3])

occupied = density > 0

height[~occupied] = 0

intensity = np.zeros_like(density)
intensity[occupied] = (
    intensity_sum[occupied] / density[occupied]
)

# Normalize density to [0, 1]
density = np.minimum(1.0, np.log1p(density) / np.log(64))

# Stack 3 channels
bev = np.stack(
    [height, density, intensity],
    axis=0
)

print("BEV shape:", bev.shape)

os.makedirs("outputs", exist_ok=True)

np.save("outputs/lidar_bev.npy", bev)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

names = ["Height", "Density", "Intensity"]

for i in range(3):
    axes[i].imshow(
        bev[i].T,
        origin="lower",
        extent=[XY_MIN, XY_MAX, XY_MIN, XY_MAX],
        cmap="viridis"
    )
    axes[i].set_title(names[i])
    axes[i].set_xlabel("X (m)")
    axes[i].set_ylabel("Y (m)")

plt.tight_layout()
plt.savefig("outputs/lidar_bev.png", dpi=150)
plt.show()