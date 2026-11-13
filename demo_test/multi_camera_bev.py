import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

DATAROOT = r'D:\bevtrain\data\nuscenes'
OUTPUT_DIR = r'D:\bevtrain\outputs'
CAMERAS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]
XY_MIN, XY_MAX, RES = -20.0, 20.0, 0.5
SIZE = int((XY_MAX - XY_MIN) / RES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    sample = nusc.sample[0]
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_calib = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    points_lidar = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_sd['token'])).points[:3]

    # Fixed LiDAR-grid indices; every camera uses the same reference frame.
    ix_all = np.floor((points_lidar[0] - XY_MIN) / RES).astype(np.int64)
    iy_all = np.floor((points_lidar[1] - XY_MIN) / RES).astype(np.int64)
    in_bev = (ix_all >= 0) & (ix_all < SIZE) & (iy_all >= 0) & (iy_all < SIZE)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder = torch.nn.Sequential(model.conv1, model.bn1, model.relu,
                                  model.maxpool, model.layer1).to(DEVICE).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    del model

    # Use CPU for accumulation; only the active camera and CNN feature map use VRAM.
    bev_sum = np.zeros((SIZE * SIZE, 64), dtype=np.float32)
    bev_count = np.zeros(SIZE * SIZE, dtype=np.int32)
    per_camera_coverage = {}
    all_camera_covered = np.zeros(SIZE * SIZE, dtype=bool)

    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for cam_name in CAMERAS:
            cam_sd = nusc.get('sample_data', sample['data'][cam_name])
            cam_calib = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
            cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])
            image_bgr = cv2.imread(nusc.get_sample_data_path(cam_sd['token']))
            if image_bgr is None:
                raise FileNotFoundError(cam_name)
            h_orig, w_orig = image_bgr.shape[:2]

            # LiDAR -> lidar ego -> global -> camera ego -> camera.
            xyz = points_lidar.copy()
            R_l = Quaternion(lidar_calib['rotation']).rotation_matrix
            t_l = np.asarray(lidar_calib['translation'])
            R_el = Quaternion(lidar_pose['rotation']).rotation_matrix
            t_el = np.asarray(lidar_pose['translation'])
            R_ec = Quaternion(cam_pose['rotation']).rotation_matrix
            t_ec = np.asarray(cam_pose['translation'])
            R_c = Quaternion(cam_calib['rotation']).rotation_matrix
            t_c = np.asarray(cam_calib['translation'])
            xyz = R_l @ xyz + t_l[:, None]
            xyz = R_el @ xyz + t_el[:, None]
            xyz = R_ec.T @ (xyz - t_ec[:, None])
            xyz = R_c.T @ (xyz - t_c[:, None])

            depth = xyz[2]
            K = np.asarray(cam_calib['camera_intrinsic'])
            uv = view_points(xyz, K, normalize=True)
            valid = (depth > 1.0) & in_bev & (uv[0] >= 0) & (uv[0] < w_orig) & (uv[1] >= 0) & (uv[1] < h_orig)
            if not np.any(valid):
                per_camera_coverage[cam_name] = 0
                print(f'{cam_name}: no valid points')
                continue

            # ResNet-18 ImageNet preprocessing, same as the previous exercise.
            rgb = cv2.cvtColor(cv2.resize(image_bgr, (800, 450)), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=torch.float32) / 255.0
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            feat = encoder((x - mean) / std)

            # grid_sample with align_corners=False and original-image pixel centers.
            u, v = uv[0, valid], uv[1, valid]
            grid = np.stack((2.0 * (u + 0.5) / w_orig - 1.0,
                             2.0 * (v + 0.5) / h_orig - 1.0), axis=-1).astype(np.float32)
            grid_t = torch.from_numpy(grid).to(DEVICE).view(1, 1, -1, 2)
            values = F.grid_sample(feat, grid_t, mode='bilinear',
                                   padding_mode='border', align_corners=False)
            values = values[0, :, 0, :].T.contiguous().cpu().numpy()

            cell_ids = ix_all[valid] * SIZE + iy_all[valid]
            np.add.at(bev_sum, cell_ids, values)
            np.add.at(bev_count, cell_ids, 1)
            unique_ids = np.unique(cell_ids)
            per_camera_coverage[cam_name] = len(unique_ids)
            all_camera_covered[unique_ids] = True
            print(f'{cam_name}: projected points={len(cell_ids)}, BEV cells={len(unique_ids)}')

            del x, feat, grid_t, values

    occupied = bev_count > 0
    bev = np.zeros_like(bev_sum)
    bev[occupied] = bev_sum[occupied] / bev_count[occupied, None]
    bev = bev.reshape(SIZE, SIZE, 64).transpose(2, 0, 1)
    np.save(os.path.join(OUTPUT_DIR, 'multi_camera_bev_features.npy'), bev)
    np.save(os.path.join(OUTPUT_DIR, 'multi_camera_bev_count.npy'), bev_count.reshape(SIZE, SIZE))

    # Coverage visualization; transpose because axis 0 is x, axis 1 is y.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.imshow((bev_count.reshape(SIZE, SIZE) > 0).T,
               origin='lower', extent=[XY_MIN, XY_MAX, XY_MIN, XY_MAX], cmap='gray_r')
    plt.xlabel('LiDAR X (m)')
    plt.ylabel('LiDAR Y (m)')
    plt.title('Six-camera BEV coverage (LiDAR-guided)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multi_camera_bev_coverage.png'), dpi=150)
    plt.close()

    print('\nCamera BEV shape:', bev.shape)
    print('Occupied cells:', int(occupied.sum()))
    print('Coverage:', f'{100 * occupied.mean():.2f}%')
    print('Device:', DEVICE)
    if DEVICE.type == 'cuda':
        print('Peak PyTorch GPU allocation (MB):',
              round(torch.cuda.max_memory_allocated() / 1024**2, 1))


if __name__ == '__main__':
    main()
