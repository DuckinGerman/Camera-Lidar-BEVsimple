import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

DATAROOT = r"D:\bevtrain\data\nuscenes"

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
    "ego_pose",
    lidar_sd["ego_pose_token"]
)
camera_pose = nusc.get(
    "ego_pose",
    camera_sd["ego_pose_token"]
)

pc = LidarPointCloud.from_file(
    nusc.get_sample_data_path(lidar_token)
)

image = cv2.imread(
    nusc.get_sample_data_path(camera_token)
)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# LiDAR -> Ego -> Global -> Camera Ego -> Camera
pc.rotate(Quaternion(lidar_calib["rotation"]).rotation_matrix)
pc.translate(np.array(lidar_calib["translation"]))

pc.rotate(Quaternion(lidar_pose["rotation"]).rotation_matrix)
pc.translate(np.array(lidar_pose["translation"]))

pc.translate(-np.array(camera_pose["translation"]))
pc.rotate(Quaternion(camera_pose["rotation"]).rotation_matrix.T)

pc.translate(-np.array(camera_calib["translation"]))
pc.rotate(Quaternion(camera_calib["rotation"]).rotation_matrix.T)

depths = pc.points[2, :]

K = np.array(camera_calib["camera_intrinsic"])

pixels = view_points(
    pc.points[:3, :],
    K,
    normalize=True
)

mask = (
    (depths > 1.0)
    & (pixels[0] >= 0)
    & (pixels[0] < image.shape[1])
    & (pixels[1] >= 0)
    & (pixels[1] < image.shape[0])
)

pixels = pixels[:, mask]
depths = depths[mask]

# Camera pixels -> RGB features
u = pixels[0].astype(int)
v = pixels[1].astype(int)

rgb_features = image[v, u].astype(np.float32) / 255.0

# Back-project pixels using measured depth
points_cam = (
    np.linalg.inv(K)
    @ np.vstack([pixels[0], pixels[1], np.ones(len(u))])
) * depths[None, :]

# Camera -> Camera Ego
R_cam = Quaternion(camera_calib["rotation"]).rotation_matrix
t_cam = np.array(camera_calib["translation"])

points_ego_cam = R_cam @ points_cam + t_cam[:, None]

# Camera Ego -> Global
R_pose_cam = Quaternion(camera_pose["rotation"]).rotation_matrix
t_pose_cam = np.array(camera_pose["translation"])

points_global = (
    R_pose_cam @ points_ego_cam
    + t_pose_cam[:, None]
)

# Global -> LiDAR Ego
R_pose_lidar = Quaternion(lidar_pose["rotation"]).rotation_matrix
t_pose_lidar = np.array(lidar_pose["translation"])

points_ego_lidar = (
    R_pose_lidar.T
    @ (points_global - t_pose_lidar[:, None])
)

# LiDAR Ego -> LiDAR
R_lidar = Quaternion(lidar_calib["rotation"]).rotation_matrix
t_lidar = np.array(lidar_calib["translation"])

points_lidar = (
    R_lidar.T
    @ (points_ego_lidar - t_lidar[:, None])
)

# Build RGB BEV
XY_MIN = -20.0
XY_MAX = 20.0
RES = 0.5
SIZE = int((XY_MAX - XY_MIN) / RES)

ix = np.floor(
    (points_lidar[0] - XY_MIN) / RES
).astype(int)

iy = np.floor(
    (points_lidar[1] - XY_MIN) / RES
).astype(int)

valid = (
    (ix >= 0) & (ix < SIZE)
    & (iy >= 0) & (iy < SIZE)
)

ix = ix[valid]
iy = iy[valid]
rgb_features = rgb_features[valid]

bev_sum = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
bev_count = np.zeros((SIZE, SIZE), dtype=np.float32)

np.add.at(bev_sum, (ix, iy), rgb_features)
np.add.at(bev_count, (ix, iy), 1)

bev_rgb = np.zeros_like(bev_sum)
occupied = bev_count > 0

bev_rgb[occupied] = (
    bev_sum[occupied] / bev_count[occupied, None]
)

os.makedirs("outputs", exist_ok=True)

np.save("outputs/camera_bev_rgb.npy", bev_rgb)

plt.figure(figsize=(7, 7))
plt.imshow(
    bev_rgb.transpose(1, 0, 2),
    origin="lower",
    extent=[XY_MIN, XY_MAX, XY_MIN, XY_MAX]
)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Camera RGB Features Projected to LiDAR BEV")
plt.tight_layout()
plt.savefig("outputs/camera_bev_rgb.png", dpi=150)
plt.show()

print("Camera BEV shape:", bev_rgb.shape)
print("Occupied cells:", occupied.sum())