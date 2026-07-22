
import os

from torch.utils.data import Dataset

from utils.transforms import SODTransform, pil_loader_mask, pil_loader_rgb

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _list_images(folder):
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(IMG_EXTENSIONS)]
    return files


def _find_matching(gt_dir, stem):
    for ext in IMG_EXTENSIONS:
        candidate = os.path.join(gt_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


class SODDataset(Dataset):
    def __init__(self, root, image_folder="images", gt_folder="gt", img_size=384, train=True, require_gt=None):
        self.root = root
        self.image_dir = os.path.join(root, image_folder)
        self.gt_dir = os.path.join(root, gt_folder) if gt_folder else None
        self.img_size = img_size
        self.train = train
        self.require_gt = train if require_gt is None else require_gt

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")

        self.image_files = _list_images(self.image_dir)
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.transform = SODTransform(img_size=img_size, train=train)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        stem = os.path.splitext(filename)[0]
        image_path = os.path.join(self.image_dir, filename)
        image = pil_loader_rgb(image_path)
        original_size = image.size[::-1]

        mask = None
        if self.gt_dir is not None:
            gt_path = _find_matching(self.gt_dir, stem)
            if gt_path is not None:
                mask = pil_loader_mask(gt_path)
            elif self.require_gt:
                raise FileNotFoundError(f"No ground-truth mask found for '{filename}' in {self.gt_dir}")

        image_t, mask_t = self.transform(image, mask)

        sample = {
            "image": image_t,
            "name": filename,
            "original_size": original_size,
        }
        if mask_t is not None:
            sample["mask"] = mask_t
        return sample
