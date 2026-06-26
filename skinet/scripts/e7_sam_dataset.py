"""
e7_sam_dataset.py

SAMKidneyDatasetE7 - PyTorch Dataset for SAM-Med2D + Kidney Adapter fine-tuning
on the E7 split (leakage-safe). Reads manifests from
e7_sam_01_build_manifest_gtbox.py.

Difference vs the previous-project dataset: a BBOX-JITTER augmentation. The
adapter is trained on GT boxes but deployed on Mask2Former boxes, which are
looser/shifted. Jittering the prompt box during training (mask target unchanged)
makes the refiner robust to that train/deploy prompt-distribution shift.

Each sample:
    image      : [3, target, target] float32, ImageNet-normalized, RGB
    mask       : [1, target, target] float32, 0/1  (SAM target)
    bbox       : [4] float32 [x1,y1,x2,y2] in target coords (the prompt)
    class_id   : int (E7 0-based)
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

PIXEL_MEAN = np.array([123.675, 116.280, 103.530], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.120, 57.375], dtype=np.float32)


class SAMKidneyDatasetE7(Dataset):
    def __init__(
        self,
        manifest_path,
        root,
        target_size: int = 256,
        augment: bool = False,
        bbox_jitter: float = 0.0,      # fraction of box side, e.g. 0.10 = +-10%
        roi_crop: bool = False,        # crop to bbox(+pad) before resize -> higher
                                       # effective resolution for thin structures
        roi_pad: float = 0.30,         # context margin each side, fraction of box side
        pixel_mean: np.ndarray = PIXEL_MEAN,
        pixel_std: np.ndarray = PIXEL_STD,
    ):
        self.root = Path(root)
        self.target_size = target_size
        self.augment = augment
        self.bbox_jitter = bbox_jitter
        self.roi_crop = roi_crop
        self.roi_pad = roi_pad
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        with open(manifest_path) as f:
            self.entries = json.load(f)

    def __len__(self):
        return len(self.entries)

    def _flip(self, image, mask, bbox, h, w):
        x1, y1, x2, y2 = bbox
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            x1, x2 = w - x2, w - x1
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
            y1, y2 = h - y2, h - y1
        return image, mask, [x1, y1, x2, y2]

    def _jitter_box(self, bbox, h, w):
        """Perturb each side by +-jitter*side. Mask (target) is NOT changed."""
        x1, y1, x2, y2 = bbox
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        j = self.bbox_jitter
        x1 += np.random.uniform(-j, j) * bw
        x2 += np.random.uniform(-j, j) * bw
        y1 += np.random.uniform(-j, j) * bh
        y2 += np.random.uniform(-j, j) * bh
        x1 = float(np.clip(x1, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        x2 = float(np.clip(x2, x1 + 1, w))
        y2 = float(np.clip(y2, y1 + 1, h))
        return [x1, y1, x2, y2]

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        image = cv2.imread(str(self.root / entry["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root / entry["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"image: {entry['image_path']}")
        if mask is None:
            raise FileNotFoundError(f"mask: {entry['mask_path']}")

        orig_h, orig_w = image.shape[:2]
        bbox = list(map(float, entry["bbox"]))

        if self.augment:
            image, mask, bbox = self._flip(image, mask, bbox, orig_h, orig_w)
            if self.bbox_jitter > 0:
                bbox = self._jitter_box(bbox, orig_h, orig_w)

        if self.roi_crop:
            x1, y1, x2, y2 = bbox
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            cx1 = int(max(0, np.floor(x1 - self.roi_pad * bw)))
            cy1 = int(max(0, np.floor(y1 - self.roi_pad * bh)))
            cx2 = int(min(orig_w, np.ceil(x2 + self.roi_pad * bw)))
            cy2 = int(min(orig_h, np.ceil(y2 + self.roi_pad * bh)))
            if cx2 - cx1 >= 2 and cy2 - cy1 >= 2:
                image = image[cy1:cy2, cx1:cx2]
                mask = mask[cy1:cy2, cx1:cx2]
                bbox = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]
                orig_h, orig_w = image.shape[:2]

        sx = self.target_size / orig_w
        sy = self.target_size / orig_h
        image = cv2.resize(image, (self.target_size, self.target_size),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.target_size, self.target_size),
                          interpolation=cv2.INTER_NEAREST)

        x1, y1, x2, y2 = bbox
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        x1 = max(0.0, min(x1, self.target_size - 1))
        y1 = max(0.0, min(y1, self.target_size - 1))
        x2 = max(x1 + 1.0, min(x2, float(self.target_size)))
        y2 = max(y1 + 1.0, min(y2, float(self.target_size)))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image - self.pixel_mean) / self.pixel_std

        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0),
            "bbox": torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
            "class_id": entry["class_id"],
            "class_name": entry["class_name"],
            "image_path": entry["image_path"],
            "mask_path": entry["mask_path"],
            "ann_id": entry["ann_id"],
        }