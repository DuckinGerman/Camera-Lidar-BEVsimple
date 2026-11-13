
import numpy as np
import torch

from nuscenes.nuscenes import NuScenes
from models.bevfusion import MultiModalBEV
from datasets.det_targets import build_targets
from losses.det_loss import DetectionLoss


def main():
    torch.manual_seed(42)
    device = "cuda"

    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=r"D:\bevtrain\data\nuscenes",
        verbose=False
    )

    sample = nusc.sample[0]

    # Load real LiDAR
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_path = nusc.get_sample_data_path(lidar_token)

    points = np.fromfile(
        lidar_path, dtype=np.float32
    ).reshape(-1, 5)[:, :4].copy()

    # Real GT targets
    targets = build_targets(nusc, sample)

    targets = {
        key: value.unsqueeze(0).to(device)
        for key, value in targets.items()
        if isinstance(value, torch.Tensor)
    }

    # Complete model
    model = MultiModalBEV().to(device).train()
    criterion = DetectionLoss()

    torch.cuda.reset_peak_memory_stats()

    # Forward
    preds = model(nusc, sample, points)

    for name, tensor in preds.items():
        print(name, tuple(tensor.shape))

    # Real detection loss
    losses = criterion(preds, targets)

    for name, value in losses.items():
        print(name, round(value.item(), 6))

    assert torch.isfinite(losses["total"]), "Non-finite loss"

    # Backward
    model.zero_grad(set_to_none=True)
    losses["total"].backward()

    modules = {
        "Depth Head": model.camera.depth_head.weight,
        "Voxel Encoder": model.lidar.voxel_encoder.mlp[0].weight,
        "Sparse CNN": model.lidar.backbone[0].weight,
        "LiDAR Compression": model.lidar.compression[0].weight,
        "Fusion": model.fusion.fusion[0].weight,
        "Detection Head": model.detection_head.heatmap.net[0].weight,
    }

    for name, parameter in modules.items():
        grad = parameter.grad
        norm = grad.norm().item() if grad is not None else float("nan")

        print(f"{name} gradient: {norm:.8g}")

        assert grad is not None, f"{name}: missing gradient"
        assert torch.isfinite(grad).all(), f"{name}: invalid gradient"
        assert norm > 0, f"{name}: zero gradient"

    print(
        "Peak GPU memory (MB):",
        round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    )

    print("Full BEVFusion detection backward OK")


if __name__ == "__main__":
    main()
