
import torch

from nuscenes.nuscenes import NuScenes
from models.detection_head import DetectionHead
from datasets.det_targets import build_targets
from losses.det_loss import DetectionLoss

device = "cuda"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=r"D:\bevtrain\data\nuscenes",
    verbose=False
)

sample = nusc.sample[0]
targets = build_targets(nusc, sample)

targets = {
    k: v.unsqueeze(0).to(device)
    for k, v in targets.items()
    if isinstance(v, torch.Tensor)
}

model = DetectionHead().to(device)
criterion = DetectionLoss()

bev = torch.randn(
    1, 64, 80, 80,
    device=device,
    requires_grad=True
)

preds = model(bev)
losses = criterion(preds, targets)

for name, value in losses.items():
    print(name, round(value.item(), 6))

losses["total"].backward()

print("BEV gradient:", bev.grad.norm().item())
print("Detection Loss backward OK")
