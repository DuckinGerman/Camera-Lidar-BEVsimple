
import os
import numpy as np
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


DATAROOT = r"D:\bevtrain\data\nuscenes"
OUTPUT_DIR = r"D:\bevtrain\outputs"

XY_MIN = -20.0
XY_MAX = 20.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_lidar_boxes(nusc, sample):
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)

    calib = nusc.get(
        "calibrated_sensor",
        sd["calibrated_sensor_token"]
    )
    pose = nusc.get(
        "ego_pose",
        sd["ego_pose_token"]
    )

    q_ge = Quaternion(pose["rotation"])
    q_el = Quaternion(calib["rotation"])

    t_ge = np.asarray(pose["translation"])
    t_el = np.asarray(calib["translation"])

    boxes = []

    for token in sample["anns"]:
        ann = nusc.get("sample_annotation", token)

        center = np.asarray(ann["translation"])
        center = q_ge.inverse.rotate(center - t_ge)
        center = q_el.inverse.rotate(center - t_el)

        q_box = (
            q_el.inverse
            * q_ge.inverse
            * Quaternion(ann["rotation"])
        )

        yaw = np.arctan2(
            q_box.rotation_matrix[1, 0],
            q_box.rotation_matrix[0, 0]
        )

        w, l, h = ann["size"]

        boxes.append({
            "center": center,
            "size": (w, l, h),
            "yaw": yaw,
            "category": ann["category_name"]
        })

    return boxes


def box_corners_2d(center, size, yaw):
    w, l, _ = size

    # nuScenes: width along local X, length along local Y
    corners = np.array([
        [-w / 2, -l / 2],
        [ w / 2, -l / 2],
        [ w / 2,  l / 2],
        [-w / 2,  l / 2],
        [-w / 2, -l / 2]
    ])

    c, s = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [c, -s],
        [s,  c]
    ])

    return corners @ R.T + center[:2]


def main():
    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=DATAROOT,
        verbose=False
    )

    sample = nusc.sample[0]

    lidar_token = sample["data"]["LIDAR_TOP"]
    path = nusc.get_sample_data_path(lidar_token)

    points = np.fromfile(
        path, dtype=np.float32
    ).reshape(-1, 5)

    mask = (
        (points[:, 0] >= XY_MIN) &
        (points[:, 0] <= XY_MAX) &
        (points[:, 1] >= XY_MIN) &
        (points[:, 1] <= XY_MAX)
    )

    points = points[mask]
    boxes = get_lidar_boxes(nusc, sample)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=0.3,
        c="gray",
        alpha=0.5
    )

    visible = 0

    for box in boxes:
        center = box["center"]

        if not (
            XY_MIN <= center[0] < XY_MAX and
            XY_MIN <= center[1] < XY_MAX
        ):
            continue

        corners = box_corners_2d(
            center,
            box["size"],
            box["yaw"]
        )

        ax.plot(
            corners[:, 0],
            corners[:, 1],
            linewidth=1.5
        )

        ax.scatter(
            center[0],
            center[1],
            s=12
        )

        visible += 1

    ax.set_xlim(XY_MIN, XY_MAX)
    ax.set_ylim(XY_MIN, XY_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("LiDAR X (m)")
    ax.set_ylabel("LiDAR Y (m)")
    ax.set_title(
        f"nuScenes LiDAR BEV + 3D GT Boxes ({visible} centers)"
    )
    ax.grid(alpha=0.2)

    output_path = os.path.join(
        OUTPUT_DIR,
        "lidar_gt_bev.png"
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print("Visible GT centers:", visible)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
