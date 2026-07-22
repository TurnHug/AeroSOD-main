import argparse
import hashlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn as nn

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mobilesam_encoder import MobileSAMv2Backbone
from models.object_aware_prompt_branch import MobileSAMv2ObjectAwarePromptEncoder
from models.s3qd_decoder import S3QD


class AeroSOD(nn.Module):
    """AeroSOD assembled from module defaults and configured weights."""

    def __init__(
        self,
        img_size,
        backbone_checkpoint,
        yolo_checkpoint,
        prompt_checkpoint,
    ):
        super().__init__()
        self.img_size = img_size
        self.yolo11n_checkpoint = str(Path(yolo_checkpoint).expanduser())

        self.backbone = MobileSAMv2Backbone(
            img_size=img_size,
            checkpoint_path=backbone_checkpoint,
        )
        in_channels = self.backbone.out_channels
        self.object_aware_prompt_encoder = MobileSAMv2ObjectAwarePromptEncoder(
            img_size=img_size,
            yolo11n_checkpoint=yolo_checkpoint,
            prompt_guided_decoder_checkpoint=prompt_checkpoint,
        )

        self._detector_parameter_count = sum(
            p.numel()
            for p in self.object_aware_prompt_encoder.yolo_detector.model.parameters()
        )
        self.decoder = S3QD(
            in_channels_list=in_channels,
            prompt_dim=self.object_aware_prompt_encoder.prompt_embed_dim,
        )

    def forward(self, x, return_aux=False):
        # x: Bx3xHxW float in the original [0,255] image range.
        prompt_out = self.object_aware_prompt_encoder(x)
        backbone_out = self.backbone(x, prompt_out)
        logits, decoder_aux = self.decoder(
            backbone_out.features,
            backbone_out.sam_embedding,
            backbone_out.coarse_logits,
            prompt_out,
            out_size=x.shape[-2:],
        )

        if not return_aux:
            return logits
        return logits, {
            "factor_weights": backbone_out.factor_weights,
            "global_logits": decoder_aux["global_logits"],
            "global_attention": decoder_aux["global_attention"],
            "proposal_mask_probs": decoder_aux["proposal_mask_probs"],
            "object_prob": decoder_aux["object_prob"],
            "prompt_out": prompt_out,
        }

    def parameter_counts(self):
        """Return registered/trainable/end-to-end counts including frozen YOLO."""
        registered = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        detector = self._detector_parameter_count
        return {
            "registered": registered,
            "detector": detector,
            "end_to_end": registered + detector,
            "trainable": trainable,
        }

    def checkpoint_metadata(self):
        """Describe the external detector that is intentionally not checkpointed."""
        path = Path(self.yolo11n_checkpoint)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        digest = None
        if path.is_file():
            sha256 = hashlib.sha256()
            with path.open("rb") as checkpoint:
                for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                    sha256.update(chunk)
            digest = sha256.hexdigest()

        try:
            ultralytics_version = version("ultralytics")
        except PackageNotFoundError:
            ultralytics_version = "unknown"

        return {
            "architecture": "AeroSOD/Lite-MFSA-v2+Global-Guided-S3QD-v2",
            "decoder_channels": self.decoder.decoder_channels,
            "external_yolo": {
                "path": str(path),
                "sha256": digest,
                "ultralytics_version": ultralytics_version,
            },
        }


