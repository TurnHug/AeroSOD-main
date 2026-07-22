

from types import SimpleNamespace

import yaml


def load_config(path="configs/default.yaml"):

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return SimpleNamespace(**cfg)


def resolve_device(device):

    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
