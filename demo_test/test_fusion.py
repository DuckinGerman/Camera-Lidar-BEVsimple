import torch
import torch.nn as nn
# from ..models.bev_fusion import BEVFusion
# from models.bev_fusion import BEVFusion


device = "cuda"

class BEVFusion(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, camera_bev, lidar_bev):
        if camera_bev.shape != lidar_bev.shape:
            raise ValueError(
                f"Shape mismatch: camera={camera_bev.shape}, "
                f"lidar={lidar_bev.shape}"
            )

        fused_bev = torch.cat(
            [camera_bev, lidar_bev], dim=1
        )

        return self.fusion(fused_bev)

    
camera_bev = torch.randn(
    1, 64, 80, 80,
    device=device,
    requires_grad=True
)

lidar_bev = torch.randn(
    1, 64, 80, 80,
    device=device,
    requires_grad=True
)

model = BEVFusion(channels=64).to(device)

fused_bev = model(camera_bev, lidar_bev)

print("Camera BEV:", tuple(camera_bev.shape))
print("LiDAR BEV:", tuple(lidar_bev.shape))
print("Fused BEV:", tuple(fused_bev.shape))

loss = fused_bev.square().mean()
loss.backward()

print("Camera gradient:", camera_bev.grad.norm().item())
print("LiDAR gradient:", lidar_bev.grad.norm().item())
print(
    "Fusion gradient:",
    model.fusion[0].weight.grad.norm().item()
)

print("BEV Fusion forward/backward OK")