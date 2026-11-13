import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

DATAROOT = r"D:\bevtrain\data\nuscenes"
OUTPUT_DIR = r"D:\bevtrain\outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 1. Load nuScenes
nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATAROOT,
    verbose=False
)

sample = nusc.sample[0]

lidar_token = sample["data"]["LIDAR_TOP"]
camera_token = sample["data"]["CAM_FRONT"]

lidar_sd = nusc.get("sample_data", lidar_token)
camera_sd = nusc.get("sample_data", camera_token)

lidar_calib = nusc.get(
    "calibrated_sensor",
    lidar_sd["calibrated_sensor_token"]
)

camera_calib = nusc.get(
    "calibrated_sensor",
    camera_sd["calibrated_sensor_token"]
)

lidar_pose = nusc.get(
    "ego_pose", lidar_sd["ego_pose_token"]
)

camera_pose = nusc.get(
    "ego_pose", camera_sd["ego_pose_token"]
)

pc = LidarPointCloud.from_file(
    nusc.get_sample_data_path(lidar_token)
)

# Preserve original LiDAR coordinates for BEV indexing
lidar_xyz = pc.points[:3].copy()

image = cv2.imread(
    nusc.get_sample_data_path(camera_token)
)

if image is None:
    raise FileNotFoundError("Camera image not found")

original_h, original_w = image.shape[:2]

# 2. Project LiDAR points to Camera
pc.rotate(
    Quaternion(lidar_calib["rotation"]).rotation_matrix
)
pc.translate(np.array(lidar_calib["translation"]))

pc.rotate(
    Quaternion(lidar_pose["rotation"]).rotation_matrix
)
pc.translate(np.array(lidar_pose["translation"]))

pc.translate(-np.array(camera_pose["translation"]))
pc.rotate(
    Quaternion(camera_pose["rotation"]).rotation_matrix.T
)

pc.translate(-np.array(camera_calib["translation"]))
pc.rotate(
    Quaternion(camera_calib["rotation"]).rotation_matrix.T
)

depths = pc.points[2]

K = np.array(camera_calib["camera_intrinsic"])

pixels = view_points(
    pc.points[:3],
    K,
    normalize=True
)

mask = (
    (depths > 1.0)
    & (pixels[0] >= 0)
    & (pixels[0] < original_w)
    & (pixels[1] >= 0)
    & (pixels[1] < original_h)
)

pixels = pixels[:, mask]
lidar_xyz = lidar_xyz[:, mask]

# 3. ResNet-18 feature extraction
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

encoder = torch.nn.Sequential(
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,
    model.layer1
).to(device).eval()

for parameter in encoder.parameters():
    parameter.requires_grad_(False)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (800, 450))

image_tensor = (
    torch.from_numpy(image)
    .permute(2, 0, 1)
    .float()
    .unsqueeze(0)
    .to(device) / 255.0
)

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=device
).view(1, 3, 1, 1)

image_tensor = (image_tensor - mean) / std

with torch.no_grad():
    feature_map = encoder(image_tensor)

print("Image feature shape:", tuple(feature_map.shape))

# 4. Sample features at projected LiDAR pixels
# grid_sample expects normalized coordinates [-1, 1].
# align_corners=False uses pixel-center normalization.

u = pixels[0]
v = pixels[1]

grid_x = 2.0 * (u + 0.5) / original_w - 1.0
grid_y = 2.0 * (v + 0.5) / original_h - 1.0

grid = np.stack([grid_x, grid_y], axis=-1)

grid_tensor = (
    torch.from_numpy(grid.astype(np.float32))
    .to(device)
    .view(1, 1, -1, 2)
)

with torch.no_grad():
    sampled = F.grid_sample(
        feature_map,
        grid_tensor,
        mode="bilinear",
        padding_mode="border",
        align_corners=False
    )

# Shape: (N, 64)
sampled_features = (
    sampled[0, :, 0, :]
    .T.contiguous()
    .cpu().numpy()
)

# 5. Map LiDAR positions to BEV grid
XY_MIN = -20.0
XY_MAX = 20.0
RES = 0.5
SIZE = 80

ix = np.floor(
    (lidar_xyz[0] - XY_MIN) / RES
).astype(int)

iy = np.floor(
    (lidar_xyz[1] - XY_MIN) / RES
).astype(int)

valid = (
    (ix >= 0) & (ix < SIZE)
    & (iy >= 0) & (iy < SIZE)
)

ix = ix[valid]
iy = iy[valid]
sampled_features = sampled_features[valid]

channels = sampled_features.shape[1]

bev_sum = np.zeros(
    (SIZE, SIZE, channels),
    dtype=np.float32
)

bev_count = np.zeros(
    (SIZE, SIZE),
    dtype=np.float32
)

np.add.at(bev_sum, (ix, iy), sampled_features)
np.add.at(bev_count, (ix, iy), 1)

bev_features = np.zeros_like(bev_sum)

occupied = bev_count > 0

bev_features[occupied] = (
    bev_sum[occupied]
    / bev_count[occupied, None]
)

# Convert HWC -> CHW
bev_features = bev_features.transpose(2, 0, 1)

np.save(
    os.path.join(OUTPUT_DIR, "camera_bev_features.npy"),
    bev_features
)

print("Camera BEV shape:", bev_features.shape)
print("Occupied cells:", int(occupied.sum()))
print("Device:", device)

if device.type == "cuda":
    print(
        "Peak GPU memory (MB):",
        round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    )