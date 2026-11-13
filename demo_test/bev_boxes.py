import os
import numpy as np
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

DATAROOT = r"D:\bevtrain\data\nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATAROOT,
    verbose=False
)

sample = nusc.sample[0]
lidar_token = sample["data"]["LIDAR_TOP"]

lidar_sd = nusc.get("sample_data", lidar_token)
lidar_calib = nusc.get(
    "calibrated_sensor",
    lidar_sd["calibrated_sensor_token"]
)
lidar_pose = nusc.get(
    "ego_pose",
    lidar_sd["ego_pose_token"]
)

pc = LidarPointCloud.from_file(
    nusc.get_sample_data_path(lidar_token)
)

points = pc.points[:3, :]

fig, ax = plt.subplots(figsize=(8, 8))

mask = (
    (points[0] >= -20) & (points[0] < 20)
    & (points[1] >= -20) & (points[1] < 20)
)

ax.scatter(
    points[0, mask],
    points[1, mask],
    s=0.2,
    c="gray"
)

for ann_token in sample["anns"]:
    ann = nusc.get("sample_annotation", ann_token)

    # Box in global coordinates
    box = nusc.get_box(ann_token)

    # Global -> Ego
    box.translate(-np.array(lidar_pose["translation"]))
    box.rotate(Quaternion(lidar_pose["rotation"]).inverse)

    # Ego -> LiDAR
    box.translate(-np.array(lidar_calib["translation"]))
    box.rotate(Quaternion(lidar_calib["rotation"]).inverse)

    # 3D box corners: shape (3, 8)
    corners = box.corners()

    # Four corners of the bottom face
    xy = corners[:2, [2, 3, 7, 6]]

    xy = np.column_stack([xy, xy[:, 0]])

    ax.plot(xy[0], xy[1], linewidth=1.5)

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_aspect("equal")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("LiDAR BEV + 3D Bounding Boxes")

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/bev_boxes.png", dpi=150)
plt.show()