

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LiteMFSAState:
    """Only the information consumed outside one Lite-MFSA stage."""

    routing: torch.Tensor
    target_logits: torch.Tensor


class TargetGuidedSCConv(nn.Module):

    def __init__(self, channels=128, alpha=0.5, squeeze_ratio=2, groups=2):
        super().__init__()
        upper_channels = int(channels * alpha)
        lower_channels = channels - upper_channels
        if upper_channels != lower_channels or channels % 2:
            raise ValueError("TargetGuidedSCConv requires an even 0.5 channel split")
        squeezed_upper = upper_channels // squeeze_ratio
        squeezed_lower = lower_channels // squeeze_ratio
        if squeezed_upper % groups:
            raise ValueError("squeezed upper channels must be divisible by groups")

        self.channels = channels
        self.spatial_norm = nn.GroupNorm(8, channels, affine=True)

        self.upper_squeeze = nn.Conv2d(
            upper_channels, squeezed_upper, kernel_size=1, bias=False,
        )
        self.lower_squeeze = nn.Conv2d(
            lower_channels, squeezed_lower, kernel_size=1, bias=False,
        )
        self.upper_group_conv = nn.Conv2d(
            squeezed_upper,
            channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.upper_point_conv = nn.Conv2d(
            squeezed_upper, channels, kernel_size=1, bias=False,
        )
        self.lower_point_conv = nn.Conv2d(
            squeezed_lower,
            channels - squeezed_lower,
            kernel_size=1,
            bias=False,
        )

    def spatial_reconstruct(self, x, target_logits):
        if target_logits.shape[1] != 1:
            raise ValueError("target_logits must contain one channel")
        if target_logits.shape[-2:] != x.shape[-2:]:
            raise ValueError("target_logits and x must have the same spatial size")

        normalized = self.spatial_norm(x)
        gamma = self.spatial_norm.weight.abs()
        gamma = gamma / gamma.sum().clamp_min(torch.finfo(gamma.dtype).eps)
        spatial_gate = torch.sigmoid(
            normalized * gamma[None, :, None, None] + target_logits
        )

        informative = spatial_gate * x
        redundant = (1.0 - spatial_gate) * x
        info_a, info_b = informative.chunk(2, dim=1)
        red_a, red_b = redundant.chunk(2, dim=1)
        return torch.cat([info_a + red_b, info_b + red_a], dim=1)

    def channel_reconstruct(self, x):
        upper, lower = x.chunk(2, dim=1)
        upper = self.upper_squeeze(upper)
        lower = self.lower_squeeze(lower)

        rich = self.upper_group_conv(upper) + self.upper_point_conv(upper)
        cheap = torch.cat([self.lower_point_conv(lower), lower], dim=1)
        candidates = torch.cat([rich, cheap], dim=1)
        weights = torch.softmax(
            F.adaptive_avg_pool2d(candidates, output_size=1), dim=1,
        )
        rich, cheap = (weights * candidates).chunk(2, dim=1)
        return rich + cheap

    def forward(self, x, target_logits):
        spatial_refined = self.spatial_reconstruct(x, target_logits)
        return self.channel_reconstruct(spatial_refined)


class MultiFactorSceneAdapter(nn.Module):

    def __init__(self, channels_list, adapter_dim=128):
        super().__init__()
        if adapter_dim != 128:
            raise ValueError("Lite-MFSA currently fixes adapter_d4im at 128")
        self.channels_list = list(channels_list)
        self.adapter_dim = adapter_dim

        self.down_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, adapter_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, adapter_dim),
                nn.GELU(),
            )
            for channels in self.channels_list
        ])
        self.up_projs = nn.ModuleList([
            nn.Conv2d(adapter_dim, channels, kernel_size=1, bias=False)
            for channels in self.channels_list
        ])
        for projection in self.up_projs:
            nn.init.normal_(projection.weight, mean=0.0, std=1e-3)

        # Shared scale unit: receptive fields of approximately 3, 5, and 7.
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(
                adapter_dim,
                adapter_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=adapter_dim,
                bias=False,
            )
            for dilation in (1, 2, 3)
        ])
        self.scale_norm = nn.GroupNorm(8, adapter_dim)

        # Shared scene unit: learn degradation from current low/high frequencies.
        self.scene_low = nn.Conv2d(
            adapter_dim, adapter_dim, kernel_size=1, bias=False,
        )
        self.scene_high = nn.Conv2d(
            adapter_dim,
            adapter_dim,
            kernel_size=3,
            padding=1,
            groups=adapter_dim,
            bias=False,
        )
        self.scene_norm = nn.GroupNorm(8, adapter_dim)

        # Shared target unit and shared TargetGuidedSCConv across all stages.
        self.target_head = nn.Conv2d(
            adapter_dim + 1, 1, kernel_size=1, bias=True,
        )
        self.target_scconv = TargetGuidedSCConv(adapter_dim)

        # Only these small routers are stage-specific.
        self.routers = nn.ModuleList([
            nn.Linear(adapter_dim, 3) for _ in self.channels_list
        ])
        self.activation = nn.GELU()

    @staticmethod
    def build_target_prior(prompt_out, image_size, output_size):
        """Rasterize valid proposal scores, using max in overlapping boxes."""
        raw_boxes = prompt_out.boxes
        dtype = raw_boxes.dtype if raw_boxes.is_floating_point() else torch.float32
        boxes = raw_boxes.to(dtype=dtype)
        scores = prompt_out.scores.to(device=boxes.device, dtype=dtype)
        box_mask = prompt_out.box_mask.to(device=boxes.device, dtype=torch.bool)
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape [B, N, 4]")
        if scores.shape != boxes.shape[:2] or box_mask.shape != boxes.shape[:2]:
            raise ValueError("scores and box_mask must have shape [B, N]")

        image_h, image_w = image_size
        output_h, output_w = output_size
        if boxes.shape[1] == 0:
            return boxes.new_zeros((boxes.shape[0], 1, output_h, output_w))

        y = (torch.arange(output_h, device=boxes.device, dtype=dtype) + 0.5)
        x = (torch.arange(output_w, device=boxes.device, dtype=dtype) + 0.5)
        y = y * (float(image_h) / output_h)
        x = x * (float(image_w) / output_w)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid_x = grid_x[None, None]
        grid_y = grid_y[None, None]

        x1 = boxes[..., 0, None, None]
        y1 = boxes[..., 1, None, None]
        x2 = boxes[..., 2, None, None]
        y2 = boxes[..., 3, None, None]
        inside = (
            (grid_x >= x1)
            & (grid_x < x2)
            & (grid_y >= y1)
            & (grid_y < y2)
            & box_mask[..., None, None]
        )
        scored_boxes = torch.where(
            inside,
            scores[..., None, None],
            torch.zeros((), device=boxes.device, dtype=dtype),
        )
        return scored_boxes.amax(dim=1, keepdim=True)

    def _scale_unit(self, z):
        scale_out = sum(convolution(z) for convolution in self.scale_convs)
        return self.activation(self.scale_norm(scale_out))

    def _scene_unit(self, z):
        low = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        high = z - low
        scene_out = self.scene_low(low) + self.scene_high(high)
        return self.activation(self.scene_norm(scene_out))

    def _target_unit(self, z, target_prior):
        prior_level = F.interpolate(
            target_prior.to(device=z.device, dtype=z.dtype),
            size=z.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        target_logits = self.target_head(torch.cat([z, prior_level], dim=1))
        return self.target_scconv(z, target_logits), target_logits

    def adapt_level(self, level_id, feature, target_prior):
        if not 0 <= level_id < len(self.channels_list):
            raise IndexError(f"invalid MFSA level_id: {level_id}")
        z = self.down_projs[level_id](feature)

        scale_out = self._scale_unit(z)
        scene_out = self._scene_unit(z)
        target_out, target_logits = self._target_unit(z, target_prior)

        descriptor = F.adaptive_avg_pool2d(z, output_size=1).flatten(1)
        routing = torch.softmax(self.routers[level_id](descriptor), dim=1)
        delta = (
            routing[:, 0, None, None, None] * scale_out
            + routing[:, 1, None, None, None] * scene_out
            + routing[:, 2, None, None, None] * target_out
        )
        adapted = feature + self.up_projs[level_id](delta)
        return adapted, LiteMFSAState(
            routing=routing,
            target_logits=target_logits,
        )

    @property
    def num_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())
