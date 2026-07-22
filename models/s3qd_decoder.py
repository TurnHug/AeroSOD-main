import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def _roi_mean(feature, boxes, box_mask, image_size):
    """Pool one feature vector for every valid proposal and pad to BxNxC."""
    batch_size, num_boxes, _ = boxes.shape
    output = feature.new_zeros(batch_size, num_boxes, feature.shape[1])
    indices = torch.nonzero(box_mask, as_tuple=False)
    if indices.numel() == 0:
        return output

    rois = torch.cat([
        indices[:, :1].to(dtype=boxes.dtype),
        boxes[box_mask].to(dtype=feature.dtype),
    ], dim=1)
    spatial_scale = feature.shape[-1] / float(image_size[1])
    pooled = roi_align(
        feature,
        rois,
        output_size=(1, 1),
        spatial_scale=spatial_scale,
        aligned=True,
    ).flatten(1)
    output[indices[:, 0], indices[:, 1]] = pooled
    return output


class _PyramidSmooth(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=1,
                groups=channels, bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class S3QD(nn.Module):
    """Decode global guidance, object prompts, and dense saliency jointly."""

    def __init__(
        self,
        in_channels_list,
        prompt_dim=256,
        decoder_channels=256,
        query_dim=96,
        query_heads=4,
    ):
        super().__init__()
        if len(in_channels_list) != 4:
            raise ValueError("S3QD expects exactly four encoder feature levels")

        self.decoder_channels = decoder_channels
        self.query_dim = query_dim

        # Component 1: merge the equal-resolution deep stages, then build a
        # compact stride-16/8/4 top-down feature pyramid.
        self.high_proj = nn.Conv2d(
            in_channels_list[0], decoder_channels, 1, bias=False,
        )
        self.mid_proj = nn.Conv2d(
            in_channels_list[1], decoder_channels, 1, bias=False,
        )
        self.deep_proj = nn.Conv2d(
            in_channels_list[2] + in_channels_list[3],
            decoder_channels,
            1,
            bias=False,
        )
        self.smooth_blocks = nn.ModuleList([
            _PyramidSmooth(decoder_channels) for _ in range(3)
        ])

        global_channels = decoder_channels // 4
        self.global_attention_head = nn.Sequential(
            nn.Conv2d(256 + 1, global_channels, 1, bias=False),
            nn.GroupNorm(8, global_channels),
            nn.GELU(),
            nn.Conv2d(global_channels, 1, 3, padding=1),
        )

        # Component 2a: one RoI feature and one prompt token form each query.
        proposal_dim = prompt_dim + decoder_channels + 6
        self.query_encoder = nn.Sequential(
            nn.Linear(proposal_dim, query_dim),
            nn.GELU(),
            nn.LayerNorm(query_dim),
        )
        self.self_attn = nn.MultiheadAttention(
            query_dim, query_heads, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.GELU(),
            nn.Linear(query_dim * 2, query_dim),
        )
        self.ffn_norm = nn.LayerNorm(query_dim)
        self.mask_kernel = nn.Linear(query_dim, decoder_channels)

        # Component 2b: keep the original dense prediction head unchanged.
        self.dense_head = nn.Sequential(
            nn.Conv2d(
                decoder_channels, decoder_channels, 3, padding=1,
                groups=decoder_channels, bias=False,
            ),
            nn.Conv2d(
                decoder_channels, decoder_channels // 2, 1, bias=False,
            ),
            nn.GroupNorm(8, decoder_channels // 2),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 2, 1, 1),
        )

        # Component 3: fuse only the prompt-aware and dense predictions. The
        # global branch has already modulated both through the shared feature.
        self.fusion_head = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    @staticmethod
    def _geometry(boxes, scores, image_size):
        height, width = image_size
        norm = boxes.new_tensor([width, height, width, height])
        coords = boxes / norm
        box_width = (boxes[..., 2] - boxes[..., 0]).clamp_min(1.0)
        box_height = (boxes[..., 3] - boxes[..., 1]).clamp_min(1.0)
        log_area = (
            box_width * box_height / float(height * width)
        ).clamp_min(1e-8).log()
        return torch.cat([
            coords, log_area[..., None], scores[..., None],
        ], dim=-1)

    @staticmethod
    def _soft_box_prior(boxes, box_mask, out_size, image_size):
        """Return one inside each box with a smooth decay outside it."""
        out_height, out_width = out_size
        image_height, image_width = image_size
        dtype, device = boxes.dtype, boxes.device
        yy = (
            torch.arange(out_height, device=device, dtype=dtype) + 0.5
        ) * image_height / out_height
        xx = (
            torch.arange(out_width, device=device, dtype=dtype) + 0.5
        ) * image_width / out_width
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        grid_x = grid_x.view(1, 1, out_height, out_width)
        grid_y = grid_y.view(1, 1, out_height, out_width)

        x1, y1, x2, y2 = [
            value[..., None, None] for value in boxes.unbind(-1)
        ]
        half_width = ((x2 - x1) * 0.5).clamp_min(1.0)
        half_height = ((y2 - y1) * 0.5).clamp_min(1.0)
        dx = (
            F.relu(x1 - grid_x) + F.relu(grid_x - x2)
        ) / half_width
        dy = (
            F.relu(y1 - grid_y) + F.relu(grid_y - y2)
        ) / half_height
        prior = torch.exp(-2.0 * (dx.square() + dy.square()))
        return prior * box_mask[..., None, None].to(dtype=dtype)

    def _pyramid(self, feats):
        if len(feats) != 4:
            raise ValueError("S3QD expects exactly four encoder feature tensors")
        if feats[2].shape[-2:] != feats[3].shape[-2:]:
            raise ValueError("the two deepest encoder features must share a resolution")

        high = self.high_proj(feats[0])
        middle = self.mid_proj(feats[1])
        deep = self.deep_proj(torch.cat([feats[2], feats[3]], dim=1))

        p16 = self.smooth_blocks[2](deep)
        p8 = self.smooth_blocks[1](
            middle + F.interpolate(
                p16, size=middle.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        )
        p4 = self.smooth_blocks[0](
            high + F.interpolate(
                p8, size=high.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        )
        return p4

    def _global_guidance(self, pixel_feature, sam_embedding, coarse_logits):
        coarse = F.interpolate(
            coarse_logits,
            size=sam_embedding.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        global_logits = self.global_attention_head(
            torch.cat([sam_embedding, coarse], dim=1)
        )
        global_attention = torch.sigmoid(F.interpolate(
            global_logits,
            size=pixel_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ))
        global_feature = pixel_feature * global_attention
        return global_feature, global_logits, global_attention

    def _refine_valid_queries(self, queries, box_mask):
        outputs = []
        for batch_index in range(queries.shape[0]):
            valid = box_mask[batch_index]
            if not valid.any():
                outputs.append(queries[batch_index])
                continue

            indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
            query = queries[batch_index, indices].unsqueeze(0)
            attended, _ = self.self_attn(
                query, query, query, need_weights=False,
            )
            query = self.attn_norm(query + attended)
            query = self.ffn_norm(query + self.ffn(query))

            full = torch.zeros_like(queries[batch_index]).index_copy(
                0, indices, query.squeeze(0),
            )
            outputs.append(torch.where(
                valid[:, None], full, queries[batch_index],
            ))
        return torch.stack(outputs, dim=0)

    def forward(
        self,
        feats,
        sam_embedding,
        coarse_logits,
        prompt_out,
        out_size,
    ):
        pixel_feature = self._pyramid(feats)
        global_feature, global_logits, global_attention = (
            self._global_guidance(
                pixel_feature, sam_embedding, coarse_logits,
            )
        )

        boxes = prompt_out.boxes
        scores = prompt_out.scores
        box_mask = prompt_out.box_mask
        roi_feature = _roi_mean(
            global_feature, boxes, box_mask, out_size,
        )
        geometry = self._geometry(boxes, scores, out_size)
        queries = self.query_encoder(torch.cat([
            prompt_out.prompt_tokens, roi_feature, geometry,
        ], dim=-1))
        queries = self._refine_valid_queries(queries, box_mask)

        kernels = self.mask_kernel(queries)
        mask_logits = torch.einsum(
            "bnc,bchw->bnhw", kernels, global_feature,
        )
        prior = self._soft_box_prior(
            boxes, box_mask, global_feature.shape[-2:], out_size,
        )
        mask_logits = mask_logits + torch.log(prior.clamp_min(1e-4))
        proposal_mask_probs = (
            torch.sigmoid(mask_logits)
            * box_mask[..., None, None].to(mask_logits.dtype)
        )

        detector_confidence = (
            scores.clamp(0.0, 1.0)
            * box_mask.to(dtype=scores.dtype)
        )
        contribution = (
            detector_confidence[..., None, None] * proposal_mask_probs
        ).clamp(0.0, 1.0 - 1e-6)
        object_prob = 1.0 - torch.exp(
            torch.log1p(-contribution).sum(dim=1, keepdim=True)
        )
        has_valid_box = box_mask.any(
            dim=1, keepdim=True,
        ).unsqueeze(-1).unsqueeze(-1)
        object_prob = torch.where(
            has_valid_box, object_prob, torch.zeros_like(object_prob),
        )
        object_map_logits = torch.where(
            has_valid_box,
            torch.logit(object_prob.clamp(1e-6, 1.0 - 1e-6)),
            torch.zeros_like(object_prob),
        )

        dense_logits = self.dense_head(global_feature)
        fused = self.fusion_head(torch.cat([
            object_map_logits, dense_logits,
        ], dim=1))
        logits = F.interpolate(
            fused, size=out_size, mode="bilinear", align_corners=False,
        )
        return logits, {
            "global_logits": global_logits,
            "global_attention": global_attention,
            "proposal_mask_probs": proposal_mask_probs,
            "object_prob": object_prob,
        }
