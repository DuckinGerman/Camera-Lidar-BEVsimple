"""
	六相机特征提取、深度预测、Lift-Splat
"""

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from pyquaternion import Quaternion

CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def transform(rotation, translation):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Quaternion(rotation).rotation_matrix
    t[:3, 3] = translation
    return t


class CameraBEV(nn.Module):
    """Six nuScenes cameras -> [1,64,80,80] in LiDAR frame, [Y,X] layout."""
    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.depth_head = nn.Conv2d(64, 8, 1)
        self.register_buffer('depths', torch.tensor([3., 6., 10., 15., 22., 30., 40., 50.]))
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # Frozen BatchNorm statistics
        return self

    @staticmethod
    def geometry(nusc, sample, camera_name, orig_h, orig_w, depths):
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_calib = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        t_l_g = np.linalg.inv(transform(lidar_pose['rotation'], lidar_pose['translation']) @
                              transform(lidar_calib['rotation'], lidar_calib['translation']))
        cam_sd = nusc.get('sample_data', sample['data'][camera_name])
        cam_calib = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
        t_l_c = (t_l_g @ transform(cam_pose['rotation'], cam_pose['translation']) @
                 transform(cam_calib['rotation'], cam_calib['translation']))
        u = (np.arange(50) + .5) * orig_w / 50
        v = (np.arange(30) + .5) * orig_h / 30
        uu, vv = np.meshgrid(u, v)
        rays = np.linalg.inv(np.asarray(cam_calib['camera_intrinsic'])) @ np.stack(
            [uu.ravel(), vv.ravel(), np.ones(1500)], axis=0)
        # Flatten depth-major to match [D,H,W] features.
        xyz = (rays[:, None, :] * depths[None, :, None]).reshape(3, -1)
        lidar_xyz = t_l_c[:3, :3] @ xyz + t_l_c[:3, 3:4]
        ix = np.floor((lidar_xyz[0] + 20.) / .5).astype(np.int64)
        iy = np.floor((lidar_xyz[1] + 20.) / .5).astype(np.int64)
        valid = (ix >= 0) & (ix < 80) & (iy >= 0) & (iy < 80)
        return (iy * 80 + ix)[valid], valid

    def forward(self, nusc, sample):
        device = self.depths.device
        bev_sum = torch.zeros((64, 6400), device=device)
        bev_weight = torch.zeros((6400,), device=device)
        for name in CAMERAS:
            path = nusc.get_sample_data_path(sample['data'][name])
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(path)
            oh, ow = image.shape[:2]
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (800, 450))
            x = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.
            x = (x - self.mean) / self.std
            with torch.no_grad():
                features = F.adaptive_avg_pool2d(self.encoder(x), (30, 50))
            probs = self.depth_head(features).softmax(1)
            lifted = (features.unsqueeze(2) * probs.unsqueeze(1)).reshape(64, -1)
            flat, valid = self.geometry(nusc, sample, name, oh, ow, self.depths.cpu().numpy())
            mask = torch.from_numpy(valid).to(device)
            indices = torch.from_numpy(flat).to(device)
            bev_sum = bev_sum.index_add(1, indices, lifted[:, mask])
            bev_weight = bev_weight.index_add(0, indices, probs.reshape(-1)[mask])
        bev = (bev_sum / bev_weight.clamp_min(1e-6).unsqueeze(0)).reshape(1, 64, 80, 80)
        return bev
