import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedStructureLoss(nn.Module):

    def __init__(self, boundary_factor=5.0, kernel_size=31):
        super().__init__()
        self.boundary_factor = boundary_factor
        self.kernel_size = kernel_size

    def forward(self, logits, targets):
        padding = self.kernel_size // 2
        local = F.avg_pool2d(
            targets, self.kernel_size, stride=1, padding=padding,
        )
        weight = 1.0 + self.boundary_factor * (local - targets).abs()

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
        wbce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1e-6)

        probs = torch.sigmoid(logits)
        inter = ((probs * targets) * weight).sum(dim=(2, 3))
        union = ((probs + targets) * weight).sum(dim=(2, 3))
        wiou = 1.0 - inter / (union - inter).clamp_min(1e-6)
        return (wbce + wiou).mean()


class AeroSODLoss(nn.Module):
    def __init__(
        self,
        sal_weight=1.0,
        global_weight=0.2,
    ):
        super().__init__()
        self.sal_weight = sal_weight
        self.global_weight = global_weight
        self.structure = WeightedStructureLoss()

    def forward(self, logits, targets, aux):
        sal = self.structure(logits, targets)
        global_logits = F.interpolate(
            aux["global_logits"], size=targets.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        global_loss = self.structure(global_logits, targets)
        total = (
            self.sal_weight * sal
            + self.global_weight * global_loss
        )
        return total, {
            "sal": sal.detach(),
            "global": global_loss.detach(),
            "total": total.detach(),
        }
