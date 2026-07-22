
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Scalar moving average (e.g., loss)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def save_checkpoint(path, model, optimizer=None, epoch=0, extra=None, scheduler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
    }
    if hasattr(model, "checkpoint_metadata"):
        state["metadata"] = model.checkpoint_metadata()
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def save_model_weights(path, model):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    map_location="cpu",
    scheduler=None,
    return_state=False,
):
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(
            "resume requires a full training checkpoint containing 'model'; "
            "use last.pth rather than a legacy weights-only file"
        )
    try:
        model.load_state_dict(state["model"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint architecture is incompatible with the current "
            "Lite-MFSA v2 / S3QD v2 model"
        ) from exc
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state if return_state else state.get("epoch", 0)
