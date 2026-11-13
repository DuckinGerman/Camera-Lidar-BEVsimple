import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# One pixel feature: C=2
feature = torch.tensor([0.2, 0.8], device=device)

# Depth bins and probabilities
depths = torch.tensor([5., 10., 15.], device=device)
probs = torch.tensor([0.1, 0.7, 0.2], device=device)

# Camera intrinsics
fx, fy = 800., 800.
cx, cy = 640., 360.
u, v = 840., 360.

# Lift
lifted = probs[:, None] * feature[None, :]

# Back-project to 3D camera coordinates
X = (u - cx) / fx * depths
Y = (v - cy) / fy * depths
Z = depths

points_cam = torch.stack([X, Y, Z], dim=1)

print("Lifted features:\n", lifted)
print("Camera 3D points:\n", points_cam)

# Toy coordinate transform:
# Camera (X right, Y down, Z forward)
# -> Ego (X forward, Y left, Z up)
points_ego = torch.stack([
    Z,
    -X,
    -Y
], dim=1)

# BEV grid
grid_size = 80
resolution = 0.5
xy_min = -20.

bev = torch.zeros((2, grid_size, grid_size), device=device)

ix = torch.floor(
    (points_ego[:, 0] - xy_min) / resolution
).long()

iy = torch.floor(
    (points_ego[:, 1] - xy_min) / resolution
).long()

valid = (
    (ix >= 0) & (ix < grid_size)
    & (iy >= 0) & (iy < grid_size)
)

for k in torch.where(valid)[0]:
    bev[:, ix[k], iy[k]] += lifted[k]

print("BEV shape:", bev.shape)
print("Occupied cells:", (bev.abs().sum(dim=0) > 0).sum().item())
print("BEV at 10m:", bev[:, 60, 35].cpu().numpy())