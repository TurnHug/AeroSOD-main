import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from models.mobilesamv2_utils import (
    MOBILESAMV2_ROOT,
    import_prompt_encoder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ULTRALYTICS_CONFIG_ROOT = PROJECT_ROOT / ".ultralytics"
_ULTRALYTICS_CONFIG_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ULTRALYTICS_CONFIG_ROOT))
_MPLCONFIG_DIR = _ULTRALYTICS_CONFIG_ROOT / "matplotlib"
_MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG_DIR))
PROMPT_GUIDED_DECODER_URL = (
    "https://huggingface.co/RogerQi/MobileSAMV2/resolve/main/Prompt_guided_Mask_Decoder.pt?download=true"
)


def _resolve_checkpoint_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _ensure_checkpoint(path, url, name, download_if_missing):
    dst = _resolve_checkpoint_path(path)
    if dst.is_file() and dst.stat().st_size > 0:
        return str(dst)

    if not download_if_missing:
        raise FileNotFoundError(f"未找到 {name} 权重: {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        if tmp.exists():
            tmp.unlink()
        print(f"[MobileSAMv2] Downloading {name} weights from {url} to {dst}")
        request = urllib.request.Request(url, headers={"User-Agent": "UAV-SOD/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response, open(tmp, "wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty")
        os.replace(tmp, dst)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Failed to automatically download {name} weights.\n"
            f"URL: {url}\nTarget path: {dst}\nReason: {exc}\n"
            "Please check the network/proxy, or manually download the file and place it in the target path."
        ) from exc
    return str(dst)


def _load_yolo11n(checkpoint_path, download_if_missing):
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    if not checkpoint_path.is_file() and not download_if_missing:
        raise FileNotFoundError(f"YOLOv11n weights not found at {checkpoint_path}")

    try:
        # 必须在 import_prompt_encoder() 之前导入，避免被 MobileSAMv2 内置的旧
        # Ultralytics fork 覆盖，从而确保这里使用支持 YOLOv11 的官方运行时。
        from ultralytics import YOLO

        if not checkpoint_path.is_file():
            print(f"[YOLOv11n] Downloading official weights to {checkpoint_path}")
        detector = YOLO(str(checkpoint_path))
    except Exception as exc:
        raise RuntimeError(
                f"Failed to load YOLOv11n weights: {checkpoint_path}\n"
            f"Reason: {exc}\n"
            "Please check the network, or manually download the official yolo11n.pt and place it in the target path."
        ) from exc

    if getattr(detector, "task", None) != "detect":
        raise ValueError(f"YOLOv11n must be a detect weights, current task is: {detector.task}")
    return detector


@dataclass
class ObjectAwarePromptOutput:
    """YOLOv11n + PromptEncoder output (pad to fixed length within batch)."""

    boxes: torch.Tensor
    scores: torch.Tensor
    box_mask: torch.Tensor
    prompt_tokens: torch.Tensor
    num_boxes: torch.Tensor

    def detach(self):
        return ObjectAwarePromptOutput(
            boxes=self.boxes.detach(),
            scores=self.scores.detach(),
            box_mask=self.box_mask.detach(),
            prompt_tokens=self.prompt_tokens.detach(),
            num_boxes=self.num_boxes.detach(),
        )


def _pool_box_sparse_embeddings(sparse):
    """SAM box two corner tokens mean pooling to a single vector."""
    if sparse.ndim == 3 and sparse.shape[1] == 2:
        return sparse.mean(dim=1)
    if sparse.ndim == 2:
        return sparse
    raise ValueError(f"unexpected sparse shape: {tuple(sparse.shape)}")


class MobileSAMv2ObjectAwarePromptEncoder(nn.Module):
    """Frozen ``YOLOv11n`` detector combined with MobileSAMv2 ``PromptEncoder``."""

    def __init__(
        self,
        img_size,
        yolo11n_checkpoint,
        prompt_guided_decoder_checkpoint,
    ):
        super().__init__()
        self.img_size = img_size
        self.max_boxes = 15
        self.download_if_missing = True
        self.prompt_embed_dim = 256

        embed_h = img_size // 16
        embed_w = img_size // 16

        object.__setattr__(self, "yolo_detector", _load_yolo11n(
            yolo11n_checkpoint, self.download_if_missing
        ))

        PromptEncoder = import_prompt_encoder()
        self.prompt_encoder = PromptEncoder(
            embed_dim=self.prompt_embed_dim,
            image_embedding_size=(embed_h, embed_w),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )

        self._load_prompt_guided_decoder_weights(prompt_guided_decoder_checkpoint)
        self._freeze_all()

    def _load_prompt_guided_decoder_weights(self, checkpoint_path):
        """Load weights from official PromptGuidedDecoder for ``PromtEncoder``."""
        checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
        if not checkpoint_path.is_file():
            alt = MOBILESAMV2_ROOT / "PromptGuidedDecoder" / "Prompt_guided_Mask_Decoder.pt"
            if alt.is_file():
                checkpoint_path = alt
            else:
                checkpoint_path = Path(_ensure_checkpoint(
                    checkpoint_path,
                    PROMPT_GUIDED_DECODER_URL,
                    "PromptGuidedDecoder",
                    self.download_if_missing,
                ))
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        self.prompt_encoder.load_state_dict(state["PromtEncoder"], strict=True)

    def _freeze_all(self):
        self.prompt_encoder.eval()
        self.yolo_detector.model.eval()
        for p in self.prompt_encoder.parameters():
            p.requires_grad_(False)
        for p in self.yolo_detector.model.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.prompt_encoder.eval()
        self.yolo_detector.model.eval()
        return self

    def _apply(self, fn):
        """Synchronize the frozen detector with the parent AeroSOD.to(device)."""
        module = super()._apply(fn)
        if hasattr(self, "yolo_detector"):
            self.yolo_detector.model.to(next(self.prompt_encoder.parameters()).device)
        return module

    @torch.no_grad()
    def _detect_boxes_batch(self, images, device):
        """Run YOLO once for a BCHW tensor and keep results on device."""
        source = images.detach().float().clamp(0, 255) / 255.0
        results = self.yolo_detector.predict(
            source=source,
            device=str(device),conf=0.2,
            verbose=False,
        )

        boxes_list, scores_list = [], []
        for i in range(images.shape[0]):
            result = results[i] if results and i < len(results) else None
            if result is None or result.boxes is None or len(result.boxes) == 0:
                boxes_list.append(torch.empty(0, 4, device=device))
                scores_list.append(torch.empty(0, device=device))
                continue
            boxes_list.append(result.boxes.xyxy.to(device=device, dtype=torch.float32))
            scores_list.append(result.boxes.conf.to(device=device, dtype=torch.float32))
        return boxes_list, scores_list

    @torch.no_grad()
    def _encode_boxes(self, boxes):
        """SAM PromptEncoder encode detected boxes."""
        sparse, _ = self.prompt_encoder(points=None, boxes=boxes, masks=None)
        return _pool_box_sparse_embeddings(sparse)

    def _pad_batch(self, boxes_list, scores_list, tokens_list, device):
        b = len(boxes_list)
        n_max = self.max_boxes
        d = self.prompt_embed_dim

        boxes = torch.zeros(b, n_max, 4, device=device)
        scores = torch.zeros(b, n_max, device=device)
        prompt_tokens = torch.zeros(b, n_max, d, device=device)
        box_mask = torch.zeros(b, n_max, dtype=torch.bool, device=device)
        num_boxes = torch.zeros(b, dtype=torch.long, device=device)

        for i, (bx, sc, tk) in enumerate(zip(boxes_list, scores_list, tokens_list)):
            n = min(bx.shape[0], n_max)
            if n == 0:
                continue
            if bx.shape[0] > n_max:
                order = torch.argsort(sc, descending=True)[:n_max]
                bx, sc, tk = bx[order], sc[order], tk[order]
                n = n_max
            boxes[i, :n] = bx[:n]
            scores[i, :n] = sc[:n]
            prompt_tokens[i, :n] = tk[:n]
            box_mask[i, :n] = True
            num_boxes[i] = n

        return ObjectAwarePromptOutput(
            boxes=boxes, scores=scores, box_mask=box_mask,
            prompt_tokens=prompt_tokens, num_boxes=num_boxes,
        )

    @torch.no_grad()
    def forward(self, images):
        if images.dtype != torch.float32:
            images = images.float()
        device = images.device
        boxes_list, scores_list = self._detect_boxes_batch(images, device)
        counts = [boxes.shape[0] for boxes in boxes_list]
        if sum(counts) == 0:
            tokens_list = [
                torch.empty(0, self.prompt_embed_dim, device=device)
                for _ in boxes_list
            ]
        else:
            # Encode every detected box in one PromptEncoder call.  This keeps
            # the whole prompt path on device and avoids a per-image encoder loop.
            flat_tokens = self._encode_boxes(torch.cat(boxes_list, dim=0))
            tokens_list = list(flat_tokens.split(counts, dim=0))

        return self._pad_batch(boxes_list, scores_list, tokens_list, device)
