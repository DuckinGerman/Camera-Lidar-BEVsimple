from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import numpy as np
import cv2
import os

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

lidar_path = nusc.get_sample_data_path(lidar_token)
camera_path = nusc.get_sample_data_path(camera_token)

pc = LidarPointCloud.from_file(lidar_path)
image = cv2.imread(camera_path)

# LiDAR sensor -> Ego at LiDAR timestamp
lidar_calib = nusc.get(
    "calibrated_sensor", lidar_sd["calibrated_sensor_token"]
)
pc.rotate(Quaternion(lidar_calib["rotation"]).rotation_matrix)
pc.translate(np.array(lidar_calib["translation"]))

# Ego at LiDAR timestamp -> Global
lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
pc.rotate(Quaternion(lidar_pose["rotation"]).rotation_matrix)
pc.translate(np.array(lidar_pose["translation"]))

# Global -> Ego at Camera timestamp
camera_pose = nusc.get("ego_pose", camera_sd["ego_pose_token"])
pc.translate(-np.array(camera_pose["translation"]))
pc.rotate(Quaternion(camera_pose["rotation"]).rotation_matrix.T)

# Ego -> Camera sensor
camera_calib = nusc.get(
    "calibrated_sensor", camera_sd["calibrated_sensor_token"]
)
pc.translate(-np.array(camera_calib["translation"]))
pc.rotate(Quaternion(camera_calib["rotation"]).rotation_matrix.T)

# Project to image
depths = pc.points[2, :]

points = view_points(
    pc.points[:3, :],
    np.array(camera_calib["camera_intrinsic"]),
    normalize=True
)

mask = (
    (depths > 1.0)
    & (points[0, :] > 0)
    & (points[0, :] < image.shape[1])
    & (points[1, :] > 0)
    & (points[1, :] < image.shape[0])
)

points = points[:, mask]
depths = depths[mask]

# Draw points
for i in range(points.shape[1]):
    u, v = int(points[0, i]), int(points[1, i])

    depth = depths[i]
    color_value = int(np.clip(255 * (1 - depth / 60), 0, 255))

    cv2.circle(
        image,
        (u, v),
        2,
        (0, color_value, 255 - color_value),
        -1
    )

os.makedirs("outputs", exist_ok=True)

cv2.imwrite("outputs/lidar_camera_projection.jpg", image)

print("Projected points:", points.shape[1])
print("Saved: outputs/lidar_camera_projection.jpg")