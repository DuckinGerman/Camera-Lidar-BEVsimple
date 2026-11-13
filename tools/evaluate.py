
import os
import random
import numpy as np
import torch

if os.name == "nt":
    torch_lib = os.path.join(
        os.path.dirname(torch.__file__), "lib"
    )
    os.add_dll_directory(torch_lib)

from nuscenes.nuscenes import NuScenes

from models.bevfusion import MultiModalBEV
from models.box_decoder import decode_boxes
from datasets.det_targets import category_to_class
from datasets.nuscenes_dataset import (
    find_available_samples,
    split_by_scene,
)
from tools.visualize import get_lidar_boxes


DATAROOT = r"D:\bevtrain\data\nuscenes"
CKPT = r"D:\bevtrain\outputs\multi_train\best.pt"

DEVICE = "cuda"
MAX_VAL_SAMPLES = 30
SCORE_THRESHOLDS = [0.1, 0.3, 0.5]
MATCH_DISTANCE = 1.0


def match_boxes(pred_boxes, gt_boxes, distance_thr=1.0):
    matched_gt = set()
    tp = 0
    distances = []

    for pred in sorted(
        pred_boxes,
        key=lambda b: b["score"],
        reverse=True
    ):
        best_idx = None
        best_dist = distance_thr

        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            if pred["class_id"] != gt["class_id"]:
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

    return (
        tp,
        len(pred_boxes) - tp,
        len(gt_boxes) - tp,
        distances
    )


def get_gt(nusc, sample):
    gt_boxes = []

    for box in get_lidar_boxes(nusc, sample):
        cls = category_to_class(box["category"])

        if cls is None:
            continue

        x, y = box["center"][:2]

        if not (-20 <= x < 20 and -20 <= y < 20):
            continue

        gt_boxes.append({
            "center": box["center"],
            "class_id": cls
        })

    return gt_boxes


def main():
    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot=DATAROOT,
        verbose=False
    )

    available = find_available_samples(nusc)

    _, val_samples = split_by_scene(
        available,
        seed=42
    )

    random.Random(43).shuffle(val_samples)
    val_samples = val_samples[:MAX_VAL_SAMPLES]

    print("Validation samples:", len(val_samples))

    model = MultiModalBEV().to(DEVICE)

    checkpoint = torch.load(
        CKPT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    results = {
        thr: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "distances": []
        }
        for thr in SCORE_THRESHOLDS
    }

    for idx, sample in enumerate(val_samples):
        lidar_path = nusc.get_sample_data_path(
            sample["data"]["LIDAR_TOP"]
        )

        points = np.fromfile(
            lidar_path,
            dtype=np.float32
        ).reshape(-1, 5)[:, :4].copy()

        with torch.inference_mode():
            preds = model(nusc, sample, points)

        all_boxes = decode_boxes(
            preds,
            score_thr=min(SCORE_THRESHOLDS),
            topk=500
        )

        gt_boxes = get_gt(nusc, sample)

        for thr in SCORE_THRESHOLDS:
            filtered = [
                b for b in all_boxes
                if b["score"] >= thr
            ]

            tp, fp, fn, distances = match_boxes(
                filtered,
                gt_boxes,
                MATCH_DISTANCE
            )

            results[thr]["tp"] += tp
            results[thr]["fp"] += fp
            results[thr]["fn"] += fn
            results[thr]["distances"].extend(distances)

        print(
            f"Evaluated {idx + 1}/{len(val_samples)}",
            flush=True
        )

    print("\nEvaluation Results")
    print("-" * 75)

    for thr, result in results.items():
        tp = result["tp"]
        fp = result["fp"]
        fn = result["fn"]

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        mean_error = (
            np.mean(result["distances"])
            if result["distances"]
            else float("nan")
        )

        print(
            f"Threshold {thr:.1f} | "
            f"TP {tp} | FP {fp} | FN {fn} | "
            f"Precision {precision:.4f} | "
            f"Recall {recall:.4f} | "
            f"Center Error {mean_error:.3f} m"
        )


if __name__ == "__main__":
    main()
