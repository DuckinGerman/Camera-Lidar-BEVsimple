
from nuscenes.nuscenes import NuScenes
from datasets.det_targets import build_targets

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=r"D:\bevtrain\data\nuscenes",
    verbose=False
)

sample = nusc.sample[0]
targets = build_targets(nusc, sample)

import torch

heatmap = targets["heatmap"]
mask = targets["mask"]

num_centers = int((heatmap == 1).sum().item())
num_regression = int(mask.sum().item())

print("GT heatmap centers:", num_centers)
print("Regression locations:", num_regression)
print("Potential center collisions:", num_centers - num_regression)

assert torch.isfinite(heatmap).all()

for name, value in targets.items():
    if hasattr(value, "shape"):
        print(name, tuple(value.shape))

print("Valid objects:", targets["num_objects"])
print("Heatmap centers:", int((targets["heatmap"] > 0).sum()))
print("Regression centers:", int(targets["mask"].sum()))
