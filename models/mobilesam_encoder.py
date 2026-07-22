import os
import urllib.request
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.mfsa import MultiFactorSceneAdapter
from models.tiny_vit import build_tiny_vit_5m

MOBILE_SAM_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)


@dataclass
class BackboneOutput:
    features: list
    sam_embedding: torch.Tensor
    factor_weights: torch.Tensor
    coarse_logits: torch.Tensor


def download_mobile_sam_weights(dst_path, url=MOBILE_SAM_URL):
    if os.path.isfile(dst_path):
        return dst_path
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    print(f"[mobilesam_encoder] Downloading weights from {url} to {dst_path}")
    urllib.request.urlretrieve(url, dst_path)
    return dst_path


def load_tiny_vit_encoder_weights(tiny_vit, checkpoint_path, verbose=True):
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw and not any(k.startswith("image_encoder") for k in raw):
        raw = raw["model"]

    if any(k.startswith("image_encoder.") for k in raw.keys()):
        encoder_sd = {k[len("image_encoder."):]: v for k, v in raw.items() if k.startswith("image_encoder.")}
    else:
        encoder_sd = raw

    missing, unexpected = tiny_vit.load_state_dict(encoder_sd, strict=True)
    if verbose:
        print(f"[mobilesam_encoder] Loaded weights from {checkpoint_path}")
        if missing:
            print(f"  missing ({len(missing)}): {missing[:4]}...")
        if unexpected:
            print(f"  unexpected ({len(unexpected)}): {unexpected[:4]}...")


class MobileSAMv2Backbone(nn.Module):
    """TinyViT multi-scale feature extractor."""

    out_strides = (4, 8, 16, 16)

    def __init__(self, img_size=384, checkpoint_path="weights/mobilesamv2/mobile_sam.pt", download_if_missing=True):
        super().__init__()
        self.img_size = img_size

        self.tiny_vit = build_tiny_vit_5m(img_size=img_size)
        self.out_channels = list(self.tiny_vit.embed_dims)

        if checkpoint_path:
            if download_if_missing and not os.path.isfile(checkpoint_path):
                download_mobile_sam_weights(checkpoint_path)
            if os.path.isfile(checkpoint_path):
                load_tiny_vit_encoder_weights(self.tiny_vit, checkpoint_path)
            else:
                print(f"[mobilesam_encoder] Warning: weights not found at {checkpoint_path}")

        self.tiny_vit.norm_head = nn.Identity()
        self.tiny_vit.head = nn.Identity()
        # Attach MFSA only after strict MobileSAM weight loading: it is now
        # an intrinsic part of the multi-scale TinyViT encoder path.
        self.tiny_vit.mfsa = MultiFactorSceneAdapter(self.out_channels)

        self.register_buffer("pixel_mean", torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(PIXEL_STD).view(1, 3, 1, 1), persistent=False)

    def normalize(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def forward(self, x, prompt_out):
        image_size = x.shape[-2:]
        target_prior = self.tiny_vit.mfsa.build_target_prior(
            prompt_out,
            image_size=image_size,
            output_size=(image_size[0] // 4, image_size[1] // 4),
        )
        x = self.normalize(x)
        feats, states = self.tiny_vit.forward_multiscale(x, target_prior)
        sam_embedding = self.tiny_vit.neck(feats[-1])
        return BackboneOutput(
            features=feats,
            sam_embedding=sam_embedding,
            factor_weights=torch.stack([state.routing for state in states], dim=1),
            coarse_logits=states[-1].target_logits,
        )
