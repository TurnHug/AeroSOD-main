
import random

import numpy as np
from PIL import Image
import torch


def pil_loader_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def pil_loader_mask(path):
    with Image.open(path) as img:
        return img.convert("L")


def image_to_tensor(img):
    arr = np.asarray(img, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def mask_to_tensor(mask):
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).contiguous()
    return tensor


class SODTransform:
    """Scale to img_size, horizontally flip during training."""

    def __init__(self, img_size=384, train=True, hflip_prob=0.5):
        self.img_size = img_size
        self.train = train
        self.hflip_prob = hflip_prob

    def __call__(self, image, mask=None):
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.train and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image_t = image_to_tensor(image)
        mask_t = mask_to_tensor(mask) if mask is not None else None
        return image_t, mask_t


