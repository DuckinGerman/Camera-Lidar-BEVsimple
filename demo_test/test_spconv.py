
import os
import torch

torch_lib = os.path.join(
    os.path.dirname(torch.__file__), "lib"
)
dll_handle = os.add_dll_directory(torch_lib)

import spconv.pytorch as spconv

device = "cuda"

# 1000 points distributed inside a 3D voxel grid
N = 1000
spatial_shape = [20, 80, 80]

# Sparse voxel coordinates: [batch, z, y, x]
indices = torch.stack([
    torch.zeros(N, dtype=torch.int32),
    torch.randint(0, 20, (N,), dtype=torch.int32),
    torch.randint(0, 80, (N,), dtype=torch.int32),
    torch.randint(0, 80, (N,), dtype=torch.int32),
], dim=1)

# Remove duplicate voxel coordinates
indices = torch.unique(indices, dim=0).to(device)

features = torch.randn(
    indices.shape[0], 4, device=device
)

x = spconv.SparseConvTensor(
    features=features,
    indices=indices,
    spatial_shape=spatial_shape,
    batch_size=1
)

model = spconv.SparseSequential(
    spconv.SubMConv3d(
        4, 16, kernel_size=3,
        padding=1, bias=False
    ),
    torch.nn.BatchNorm1d(16),
    torch.nn.ReLU()
).to(device).eval()

with torch.no_grad():
    y = model(x)

torch.cuda.synchronize()

print("Input features:", tuple(x.features.shape))
print("Output features:", tuple(y.features.shape))
print("Output spatial shape:", y.spatial_shape)
print(
    "Peak GPU memory (MB):",
    round(torch.cuda.max_memory_allocated() / 1024**2, 1)
)
print("Sparse 3D CNN forward OK")
