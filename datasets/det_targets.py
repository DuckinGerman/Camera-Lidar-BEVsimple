
import numpy as np
import torch

from pyquaternion import Quaternion

CLASSES = [
    "car", "truck", "construction_vehicle", "bus",
    "trailer", "barrier", "motorcycle", "bicycle",
    "pedestrian", "traffic_cone"
]

XY_MIN = -20.0
RES = 0.5
SIZE = 80


def category_to_class(category):
    # nuScenes fine-grained categories -> detection classes
    if category.startswith("vehicle.car"):
        return 0
    if category.startswith("vehicle.truck"):
        return 1
    if category.startswith("vehicle.construction"):
        return 2
    if category.startswith("vehicle.bus"):
        return 3
    if category.startswith("vehicle.trailer"):
        return 4
    if category.startswith("movable_object.barrier"):
        return 5
    if category.startswith("vehicle.motorcycle"):
        return 6
    if category.startswith("vehicle.bicycle"):
        return 7
    if category.startswith("human.pedestrian"):
        return 8
    if category.startswith("movable_object.trafficcone"):
        return 9
    return None


def build_targets(nusc, sample):
    heatmap = torch.zeros(10, SIZE, SIZE)
    offset = torch.zeros(2, SIZE, SIZE)
    height = torch.zeros(1, SIZE, SIZE)
    dimensions = torch.zeros(3, SIZE, SIZE)
    rotation = torch.zeros(2, SIZE, SIZE)
    mask = torch.zeros(1, SIZE, SIZE)

    lidar_sd = nusc.get(
        "sample_data", sample["data"]["LIDAR_TOP"]
    )
    calib = nusc.get(
        "calibrated_sensor",
        lidar_sd["calibrated_sensor_token"]
    )
    pose = nusc.get(
        "ego_pose", lidar_sd["ego_pose_token"]
    )

    q_global_ego = Quaternion(pose["rotation"])
    q_ego_lidar = Quaternion(calib["rotation"])

    t_global_ego = np.asarray(pose["translation"])
    t_ego_lidar = np.asarray(calib["translation"])

    count = 0

    for token in sample["anns"]:
        ann = nusc.get("sample_annotation", token)
        cls = category_to_class(ann["category_name"])

        if cls is None:
            continue

        # Global -> Ego -> LiDAR
        center = np.asarray(ann["translation"])
        center = q_global_ego.inverse.rotate(
            center - t_global_ego
        )
        center = q_ego_lidar.inverse.rotate(
            center - t_ego_lidar
        )

        gx = (center[0] - XY_MIN) / RES
        gy = (center[1] - XY_MIN) / RES

        ix = int(np.floor(gx))
        iy = int(np.floor(gy))

        if not (0 <= ix < SIZE and 0 <= iy < SIZE):
            continue

        # nuScenes size: [width, length, height]
        w, l, h = ann["size"]

        # Box orientation in LiDAR coordinates
        q_box = (
            q_ego_lidar.inverse
            * q_global_ego.inverse
            * Quaternion(ann["rotation"])
        )

        yaw = np.arctan2(
            q_box.rotation_matrix[1, 0],
            q_box.rotation_matrix[0, 0]
        )

        # Tensor spatial indexing: [Y, X]
        # heatmap[cls, iy, ix] = 1.0
        radius = gaussian_radius(w / RES, l / RES)

        draw_gaussian(
            heatmap[cls],
            ix,
            iy,
            radius
        )
        
        offset[:, iy, ix] = torch.tensor(
            [gx - ix, gy - iy], dtype=torch.float32
        )
        height[0, iy, ix] = float(center[2])
        dimensions[:, iy, ix] = torch.log(
            torch.tensor([w, l, h], dtype=torch.float32)
        )
        rotation[:, iy, ix] = torch.tensor(
            [np.sin(yaw), np.cos(yaw)],
            dtype=torch.float32
        )
        mask[0, iy, ix] = 1.0
        count += 1

    return {
        "heatmap": heatmap,
        "offset": offset,
        "height": height,
        "dimensions": dimensions,
        "rotation": rotation,
        "mask": mask,
        "num_objects": count
    }



def gaussian_radius(width, length, min_overlap=0.1):
    a = 1
    b = width + length
    c = width * length * (1 - min_overlap) / (1 + min_overlap)

    radius = (b - np.sqrt(max(0.0, b*b - 4*a*c))) / 2
    return max(0, int(radius))


def draw_gaussian(heatmap, center_x, center_y, radius):
    diameter = 2 * radius + 1
    sigma = diameter / 6

    x = torch.arange(diameter, dtype=torch.float32) - radius
    y = torch.arange(diameter, dtype=torch.float32) - radius

    yy, xx = torch.meshgrid(y, x, indexing="ij")
    gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    height, width = heatmap.shape

    left = min(center_x, radius)
    right = min(width - center_x - 1, radius)
    top = min(center_y, radius)
    bottom = min(height - center_y - 1, radius)

    patch = heatmap[
        center_y - top:center_y + bottom + 1,
        center_x - left:center_x + right + 1
    ]

    kernel = gaussian[
        radius - top:radius + bottom + 1,
        radius - left:radius + right + 1
    ]

    torch.maximum(patch, kernel, out=patch)
