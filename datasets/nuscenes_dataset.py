
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.det_targets import build_targets


SENSORS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "LIDAR_TOP",
]


def find_available_samples(nusc):
    """Only keep samples with all seven sensor files."""

    available = []

    for sample in nusc.sample:
        complete = True

        for sensor in SENSORS:
            token = sample["data"].get(sensor)

            if token is None:
                complete = False
                break

            path = nusc.get_sample_data_path(token)

            if not os.path.isfile(path):
                complete = False
                break

        if complete:
            available.append(sample)

    return available


def split_by_scene(samples, val_ratio=0.2, seed=42):
    """Scene-level split to reduce train/val leakage."""

    scenes = sorted(set(s["scene_token"] for s in samples))

    if len(scenes) < 2:
        raise ValueError(
            "Need at least two complete scenes for train/val split."
        )

    rng = random.Random(seed)
    rng.shuffle(scenes)

    n_val = max(1, round(len(scenes) * val_ratio))
    n_val = min(n_val, len(scenes) - 1)

    val_scenes = set(scenes[:n_val])

    train_samples = [
        s for s in samples
        if s["scene_token"] not in val_scenes
    ]

    val_samples = [
        s for s in samples
        if s["scene_token"] in val_scenes
    ]

    return train_samples, val_samples


class NuScenesBEVDataset(Dataset):
    def __init__(self, nusc, samples):
        self.nusc = nusc
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        lidar_path = self.nusc.get_sample_data_path(
            sample["data"]["LIDAR_TOP"]
        )

        points = np.fromfile(
            lidar_path,
            dtype=np.float32
        ).reshape(-1, 5)[:, :4].copy()

        targets = build_targets(self.nusc, sample)

        return {
            "sample": sample,
            "points": points,
            "targets": targets
        }
