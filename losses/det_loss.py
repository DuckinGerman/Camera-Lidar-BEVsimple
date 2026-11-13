
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # Heatmap classification
        # heatmap_loss = F.binary_cross_entropy_with_logits(
        #     preds["heatmap"],
        #     targets["heatmap"]
        # )

        pred = torch.sigmoid(preds["heatmap"]).clamp(1e-4, 1 - 1e-4)
        gt = targets["heatmap"]

        pos = (gt == 1).float()
        neg = (gt < 1).float()
        neg_weight = (1 - gt).pow(4)

        pos_loss = (
            torch.log(pred) *
            (1 - pred).pow(2) *
            pos
        )

        neg_loss = (
            torch.log(1 - pred) *
            pred.pow(2) *
            neg_weight *
            neg
        )

        num_pos = pos.sum()

        heatmap_loss = -(
            pos_loss.sum() + neg_loss.sum()
        ) / num_pos.clamp_min(1.0)

        # Regression only at positive centers
        mask = targets["mask"]
        num_pos = mask.sum().clamp_min(1.0)

        losses = {}

        for key in [
            "offset", "height", "dimensions", "rotation"
        ]:
            pred = preds[key]
            target = targets[key]

            loss = F.smooth_l1_loss(
                pred * mask,
                target * mask,
                reduction="sum"
            ) / num_pos

            losses[key] = loss

        regression_loss = sum(losses.values())

        total_loss = heatmap_loss + regression_loss

        return {
            "total": total_loss,
            "heatmap": heatmap_loss,
            **losses
        }
