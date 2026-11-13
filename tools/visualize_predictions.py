
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from models.bevfusion import MultiModalBEV
from models.box_decoder import decode_boxes
from datasets.det_targets import category_to_class
from tools.visualize import (
    get_lidar_boxes,
    box_corners_2d
)

DATAROOT = r"D:\bevtrain\data\nuscenes"
CKPT = r"D:\bevtrain\outputs\checkpoints\best_overfit.pt"
OUTPUT = r"D:\bevtrain\outputs\pred_vs_gt.png"


def main():
    device = "cuda"

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=DATAROOT,
        verbose=False
    )

    checkpoint = torch.load(
        CKPT, map_location=device, weights_only=False
    )

    sample = nusc.get(
        "sample", checkpoint["sample_token"]
    )

    lidar_path = nusc.get_sample_data_path(
        sample["data"]["LIDAR_TOP"]
    )

    points = np.fromfile(
        lidar_path, dtype=np.float32
    ).reshape(-1, 5)[:, :4].copy()

    model = MultiModalBEV().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.inference_mode():
        preds = model(nusc, sample, points)

    boxes = decode_boxes(
        preds,
        score_thr=0.3,
        topk=100
    )

    gt_boxes = [
        box for box in get_lidar_boxes(nusc, sample)
        if category_to_class(box["category"]) is not None
        and -20 <= box["center"][0] < 20
        and -20 <= box["center"][1] < 20
    ]

    fig, ax = plt.subplots(figsize=(10, 10))

    mask = (
        (points[:, 0] >= -20) &
        (points[:, 0] < 20) &
        (points[:, 1] >= -20) &
        (points[:, 1] < 20)
    )

    ax.scatter(
        points[mask, 0],
        points[mask, 1],
        s=0.3,
        c="gray",
        alpha=0.35
    )

    # GT: green
    for box in gt_boxes:
        corners = box_corners_2d(
            box["center"],
            box["size"],
            box["yaw"]
        )
        ax.plot(
            corners[:, 0],
            corners[:, 1],
            color="green",
            linewidth=1.5
        )

    # Predictions: red
    for box in boxes:
        corners = box_corners_2d(
            box["center"],
            box["size"],
            box["yaw"]
        )
        ax.plot(
            corners[:, 0],
            corners[:, 1],
            color="red",
            linewidth=1.2
        )

    matched_gt = set()
    tp = 0
    distances = []

    for pred in sorted(boxes, key=lambda x: x["score"], reverse=True):
        best_idx = None
        best_dist = 1.0

        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            if pred["class_id"] != category_to_class(gt["category"]):
                continue

            dist = np.linalg.norm(
                np.array(pred["center"][:2]) -
                np.array(gt["center"][:2])
            )

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            matched_gt.add(best_idx)
            tp += 1
            distances.append(best_dist)

    fp = len(boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(
        "Mean center error (m):",
        np.mean(distances) if distances else None
    )

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect("equal")
    ax.set_xlabel("LiDAR X (m)")
    ax.set_ylabel("LiDAR Y (m)")
    ax.set_title(
        f"GT (green): {len(gt_boxes)} | "
        f"Predictions (red): {len(boxes)}"
    )
    ax.grid(alpha=0.2)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.show()

    print("GT boxes:", len(gt_boxes))
    print("Predicted boxes:", len(boxes))
    print("Saved:", OUTPUT)

    for box in sorted(
        boxes, key=lambda b: b["score"], reverse=True
    )[:10]:
        print(
            box["class_name"],
            "score:", round(box["score"], 3),
            "center:", np.round(box["center"], 2)
        )


if __name__ == "__main__":
    main()
