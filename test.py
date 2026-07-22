import argparse
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.sod_dataset import SODDataset
from models.aero_sod import AeroSOD
from utils.config import load_config, resolve_device


def main():
    parser = argparse.ArgumentParser(description="Run SOD inference")
    parser.add_argument("--config", default="configs/default.yaml", help="config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.device)
    print(f"[test] device = {device}")


    test_set = SODDataset(
        root=cfg.test_root,
        image_folder=cfg.test_image_folder,
        gt_folder=cfg.test_gt_folder,
        img_size=cfg.img_size,
        train=False,
        require_gt=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    print(f"[test] test samples: {len(test_set)}")
    model = AeroSOD(
        img_size=cfg.img_size,
        backbone_checkpoint=cfg.mobilesam_checkpoint,
        yolo_checkpoint=cfg.yolo11n_checkpoint,
        prompt_checkpoint=cfg.prompt_guided_decoder_checkpoint,
    ).to(device)

    if cfg.checkpoint and os.path.isfile(cfg.checkpoint):
        state = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
        state_dict = state["model"] if "model" in state else state
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint is not compatible with the current structure: "
                f"{cfg.checkpoint}. Please retrain with a new structure."
            ) from exc
        print(f"[test] loaded trained weights from {cfg.checkpoint}")
    else:
        print(f"[test] WARNING: no trained checkpoint found at '{cfg.checkpoint}'. "
              f"Running with randomly-initialized adapter/decoder "
              f"(TinyViT is pretrained; YOLO + PromptEncoder are frozen).")

    model.eval()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- 推理 ----
    n_saved = 0
    time_sum = 0.0
    use_cuda = str(device).startswith("cuda")
    with torch.inference_mode():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            names = batch["name"]
            original_sizes = batch["original_size"]

            if use_cuda:
                torch.cuda.synchronize()
            time_start = time.time()
            logits = model(images)
            if use_cuda:
                torch.cuda.synchronize()
            time_sum += time.time() - time_start

            for i in range(logits.size(0)):
                h, w = int(original_sizes[0][i]), int(original_sizes[1][i])

                logit = F.interpolate(
                    logits[i:i + 1], size=(h, w),
                    mode="bilinear", align_corners=False,
                )
                # Preserve calibrated probabilities. Per-image min-max
                # normalization changes MAE/F-measure and is unsuitable for
                # quantitative SOD evaluation.
                prob = torch.sigmoid(logit).cpu().numpy().squeeze()
                pred_uint8 = np.rint(prob * 255.0).astype(np.uint8)

                out_name = os.path.splitext(names[i])[0] + ".png"
                imageio.imsave(os.path.join(cfg.output_dir, out_name), pred_uint8)
                n_saved += 1

    print(f"[test] saved {n_saved} saliency maps to '{cfg.output_dir}'.")
    print(f"[test] avg inference time: {time_sum / max(n_saved, 1):.4f}s | "
          f"FPS: {n_saved / max(time_sum, 1e-6):.2f}")


if __name__ == "__main__":
    main()
