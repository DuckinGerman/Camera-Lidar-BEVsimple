
import torch
import torch.nn.functional as F

from datasets.det_targets import CLASSES, XY_MIN, RES


@torch.no_grad()
def decode_boxes(preds, score_thr=0.3, topk=100):
    """
    Input: DetectionHead outputs, batch size 1
    Output: list of 3D boxes in LiDAR coordinates
    """

    heatmap = preds["heatmap"].sigmoid()
    assert heatmap.shape[0] == 1

    # Keep local peaks
    pooled = F.max_pool2d(
        heatmap, kernel_size=3, stride=1, padding=1
    )
    heatmap = heatmap * (heatmap == pooled)

    _, num_classes, H, W = heatmap.shape

    scores, indices = torch.topk(
        heatmap.reshape(-1),
        k=min(topk, heatmap.numel())
    )

    boxes = []

    for score, index in zip(scores, indices):
        if score.item() < score_thr:
            continue

        index = int(index.item())
        cls = index // (H * W)
        pos = index % (H * W)
        iy = pos // W
        ix = pos % W

        dx, dy = preds["offset"][0, :, iy, ix]
        z = preds["height"][0, 0, iy, ix]

        # Dimensions are encoded as log(w,l,h)
        w, l, h = preds["dimensions"][
            0, :, iy, ix
        ].exp()

        sin_yaw, cos_yaw = preds["rotation"][
            0, :, iy, ix
        ]

        yaw = torch.atan2(sin_yaw, cos_yaw)

        x = XY_MIN + (ix + dx.item()) * RES
        y = XY_MIN + (iy + dy.item()) * RES

        boxes.append({
            "center": [x, y, z.item()],
            "size": [w.item(), l.item(), h.item()],
            "yaw": yaw.item(),
            "class_id": cls,
            "class_name": CLASSES[cls],
            "score": score.item()
        })

    return boxes
