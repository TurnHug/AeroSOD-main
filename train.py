import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from datasets.sod_dataset import SODDataset
from models.aero_sod import AeroSOD
from utils.config import load_config, resolve_device
from utils.losses import AeroSODLoss
from utils.misc import (
    AverageMeter,
    load_checkpoint,
    save_checkpoint,
    save_model_weights,
    set_seed,
)


def build_model(cfg):
    return AeroSOD(
        img_size=cfg.img_size,
        backbone_checkpoint=cfg.mobilesam_checkpoint,
        yolo_checkpoint=cfg.yolo11n_checkpoint,
        prompt_checkpoint=cfg.prompt_guided_decoder_checkpoint,
    )


def build_loader(cfg, root, train):
    dataset = SODDataset(
        root=root,
        image_folder=cfg.train_image_folder if train else cfg.val_image_folder,
        gt_folder=cfg.train_gt_folder if train else cfg.val_gt_folder,
        img_size=cfg.img_size,
        train=train,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=str(resolve_device(cfg.device)).startswith("cuda"),
        drop_last=train,
    )
    return dataset, loader



def main():
    parser = argparse.ArgumentParser(description="Train AeroSOD")
    parser.add_argument("--config", default="configs/default.yaml", help="config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[train] device = {device}")

    train_set, train_loader = build_loader(cfg, cfg.train_root, train=True)
    print(f"[train] train samples: {len(train_set)}")
    val_loader = None
    if cfg.val_root:
        val_set, val_loader = build_loader(cfg, cfg.val_root, train=False)
        print(f"[train] val samples: {len(val_set)}")
    else:
        print(
            "[train] WARNING: val_root is empty; best.pth and validation-based "
            "model selection are disabled. Use last.pth for inference."
        )
    if len(train_loader) == 0:
        raise ValueError(
            "training loader is empty; reduce batch_size or disable drop_last"
        )

    model = build_model(cfg).to(device)
    counts = model.parameter_counts()
    print(
        f"[train] trainable params: {counts['trainable']:,} / "
        f"{counts['end_to_end']:,} (including frozen YOLO11n)"
    )

    adapter_ids = {
        id(p) for p in model.backbone.tiny_vit.mfsa.parameters()
    }
    backbone_params = [
        p for p in model.backbone.tiny_vit.parameters()
        if p.requires_grad and id(p) not in adapter_ids
    ]
    backbone_ids = {id(p) for p in backbone_params}
    module_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_ids
    ]
    backbone_lr_ratio = getattr(cfg, "backbone_lr_ratio", 0.1)
    if not 0.0 < backbone_lr_ratio <= 1.0:
        raise ValueError("backbone_lr_ratio must be in (0, 1]")
    backbone_lr = cfg.module_lr * backbone_lr_ratio
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": module_params, "lr": cfg.module_lr},
    ], weight_decay=cfg.weight_decay)
    print(
        f"[train] lr: TinyViT={backbone_lr:.2e} "
        f"({backbone_lr_ratio:.2f}x), modules={cfg.module_lr:.2e}"
    )
    criterion = AeroSODLoss(
        sal_weight=cfg.sal_weight,
        global_weight=cfg.global_weight,
    )

    module_min_lr = getattr(cfg, "module_min_lr", 1e-6)
    if not 0.0 <= module_min_lr <= cfg.module_lr:
        raise ValueError("module_min_lr must be in [0, module_lr]")
    minimum_ratio = module_min_lr / cfg.module_lr

    def lr_factor(epoch):
        if epoch < cfg.lr_warmup_epochs:
            return 0.1 + 0.9 * (epoch + 1) / max(cfg.lr_warmup_epochs, 1)
        progress = (epoch - cfg.lr_warmup_epochs) / max(
            cfg.epochs - cfg.lr_warmup_epochs - 1, 1,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        # Apply the same relative schedule so TinyViT stays at 0.1x LR.
        lr_lambda=[lr_factor, lr_factor],
    )
    start_epoch = 0
    best_mae = float("inf")
    if cfg.resume:
        resume_state = load_checkpoint(
            cfg.resume, model, optimizer, scheduler=scheduler,
            map_location=device, return_state=True,
        )
        start_epoch = resume_state.get("epoch", 0) + 1
        best_mae = float(resume_state.get("best_mae", best_mae))
        print(f"[train] resumed from {cfg.resume}, starting at epoch {start_epoch}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        meters = {
            name: AverageMeter() for name in ("sal", "global", "total")
        }
        diagnostics = {
            name: AverageMeter() for name in (
                "route_scale", "route_scene", "route_target",
                "attention", "object_prob", "fusion_object_share",
                "valid_boxes",
            )
        }
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(images, return_aux=True)

            loss, parts = criterion(logits, masks, aux)
            loss.backward()
            optimizer.step()

            for name, value in parts.items():
                meters[name].update(value.item(), n=images.size(0))
            routing = aux["factor_weights"].detach().mean(dim=(0, 1))
            for index, name in enumerate(
                ("route_scale", "route_scene", "route_target")
            ):
                diagnostics[name].update(routing[index].item(), n=images.size(0))
            diagnostics["attention"].update(
                aux["global_attention"].detach().mean().item(),
                n=images.size(0),
            )
            diagnostics["object_prob"].update(
                aux["object_prob"].detach().mean().item(),
                n=images.size(0),
            )
            fusion_weights = model.decoder.fusion_head[0].weight.detach()
            fusion_norms = fusion_weights.square().sum(dim=(0, 2, 3)).sqrt()
            diagnostics["fusion_object_share"].update(
                (
                    fusion_norms[0]
                    / fusion_norms.sum().clamp_min(torch.finfo(fusion_norms.dtype).eps)
                ).item(),
                n=images.size(0),
            )
            diagnostics["valid_boxes"].update(
                aux["prompt_out"].box_mask.sum(dim=1).float().mean().item(),
                n=images.size(0),
            )
            if (step + 1) % cfg.log_interval == 0:
                print(
                    f"[train] epoch {epoch} step {step + 1}/{len(train_loader)} "
                    f"total {meters['total'].avg:.4f} sal {meters['sal'].avg:.4f} "
                    f"global {meters['global'].avg:.4f} | route "
                    f"S/C/T={diagnostics['route_scale'].avg:.3f}/"
                    f"{diagnostics['route_scene'].avg:.3f}/"
                    f"{diagnostics['route_target'].avg:.3f} "
                    f"attention={diagnostics['attention'].avg:.3f} "
                    f"object={diagnostics['object_prob'].avg:.3f} "
                    f"fusion_obj={diagnostics['fusion_object_share'].avg:.3f} "
                    f"boxes={diagnostics['valid_boxes'].avg:.2f}"
                )

        scheduler.step()
        elapsed = time.time() - t0
        msg = (
            f"[train] epoch {epoch} done in {elapsed:.1f}s - "
            f"loss {meters['total'].avg:.4f}"
        )
        if val_loader is not None:
            val_loss, val_mae = validate(model, val_loader, criterion, device)
            msg += f" - val loss {val_loss:.4f} - val mae {val_mae:.4f}"
            if val_mae < best_mae:
                best_mae = val_mae
                save_model_weights(
                    os.path.join(cfg.save_dir, "best.pth"),
                    model,
                )
                print(f"[train] new best mae: {best_mae:.4f}, saved best.pth")
        print(msg)

        save_checkpoint(
            os.path.join(cfg.save_dir, "last.pth"),
            model, optimizer, epoch, scheduler=scheduler,
            extra={"best_mae": best_mae},
        )

    print("[train] training finished.")


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter, mae_meter = AverageMeter(), AverageMeter()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits, aux = model(images, return_aux=True)
        loss, _ = criterion(logits, masks, aux)
        mae = (torch.sigmoid(logits) - masks).abs().mean()
        loss_meter.update(loss.item(), n=images.size(0))
        mae_meter.update(mae.item(), n=images.size(0))
    model.train()
    return loss_meter.avg, mae_meter.avg


if __name__ == "__main__":
    main()
