import os
import sys
from pathlib import Path
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.bevfusion import MultiModalBEV

DATAROOT = r'D:\bevtrain\data\nuscenes'


def main():
    torch.manual_seed(42)
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    sample = nusc.sample[0]
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    path = nusc.get_sample_data_path(lidar_sd['token'])
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4].copy()
    model = MultiModalBEV().cuda().train()
    torch.cuda.reset_peak_memory_stats()
    fused, cam, lidar = model(nusc, sample, points)
    print('Camera BEV:', tuple(cam.shape))
    print('LiDAR BEV:', tuple(lidar.shape))
    print('Fused BEV:', tuple(fused.shape))
    loss = fused.square().mean()  # Connectivity test only
    loss.backward()
    grads = {
        'Depth Head': model.camera.depth_head.weight.grad,
        'Voxel Encoder': model.lidar.voxel_encoder.mlp[0].weight.grad,
        'Sparse CNN': model.lidar.backbone[0].weight.grad,
        'LiDAR Compression': model.lidar.compression[0].weight.grad,
        'Fusion': model.fusion.fusion[0].weight.grad,
    }
    for name, grad in grads.items():
        norm = float(grad.norm()) if grad is not None else float('nan')
        print(f'{name} gradient norm: {norm:.8g}')
        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0, name
    print('Peak PyTorch GPU allocation (MB):', round(torch.cuda.max_memory_allocated()/1024**2, 1))
    print('Real Camera-LiDAR BEV fusion forward/backward OK')


if __name__ == '__main__':
    main()
