"""
	调用 Camera、LiDAR、Fusion、Detection Head
"""

# from torch import nn
# from .camera_bev import CameraBEV
# from .lidar_bev import LiDARBEV
# from .bev_fusion import BEVFusion


# class MultiModalBEV(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.camera = CameraBEV()
#         self.lidar = LiDARBEV()
#         self.fusion = BEVFusion()

#     def forward(self, nusc, sample, lidar_points):
#         camera_bev = self.camera(nusc, sample)
#         lidar_bev = self.lidar(lidar_points)
#         fused_bev = self.fusion(camera_bev, lidar_bev)
#         return fused_bev, camera_bev, lidar_bev


from torch import nn

from .camera_bev import CameraBEV
from .lidar_bev import LiDARBEV
from .bev_fusion import BEVFusion
from .detection_head import DetectionHead


class MultiModalBEV(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.camera = CameraBEV()
        self.lidar = LiDARBEV()
        self.fusion = BEVFusion(channels=64)
        self.detection_head = DetectionHead(
            in_channels=64,
            num_classes=num_classes
        )

    def forward(self, nusc, sample, lidar_points):
        camera_bev = self.camera(nusc, sample)
        lidar_bev = self.lidar(lidar_points)

        fused_bev = self.fusion(camera_bev, lidar_bev)

        predictions = self.detection_head(fused_bev)

        return predictions
