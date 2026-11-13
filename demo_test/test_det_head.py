
import torch
# from models.detection_head import DetectionHead



import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x)


class DetectionHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=10):
        super().__init__()

        self.heatmap = PredictionHead(in_channels, num_classes)
        self.offset = PredictionHead(in_channels, 2)
        self.height = PredictionHead(in_channels, 1)
        self.dimensions = PredictionHead(in_channels, 3)
        self.rotation = PredictionHead(in_channels, 2)

        # Low initial probability for object centers
        nn.init.constant_(self.heatmap.net[-1].bias, -2.19)

    def forward(self, bev):
        return {
            "heatmap": self.heatmap(bev),
            "offset": self.offset(bev),
            "height": self.height(bev),
            "dimensions": self.dimensions(bev),
            "rotation": self.rotation(bev)
        }


device = "cuda"

model = DetectionHead(
    in_channels=64,
    num_classes=10
).to(device)

bev = torch.randn(
    1, 64, 80, 80,
    device=device,
    requires_grad=True
)

outputs = model(bev)

for name, value in outputs.items():
    print(name, tuple(value.shape))

loss = sum(
    value.square().mean()
    for value in outputs.values()
)

loss.backward()

print("BEV gradient:", bev.grad.norm().item())
print(
    "Heatmap Head gradient:",
    model.heatmap.net[0].weight.grad.norm().item()
)

print("3D Detection Head forward/backward OK")
