# -*- coding: utf-8 -*-
"""
src/dataset.py
--------------
Xây dựng DataLoader cho bài toán phân loại ảnh y tế.

LƯU Ý Y KHOA:
    RandomHorizontalFlip bị loại bỏ cố ý.
    Trong ảnh y tế (X-quang ngực, siêu âm, MRI, ...) việc lật ngang
    thay đổi tính đối xứng giải phẫu và có thể tạo ra mẫu không hợp lệ
    về mặt lâm sàng, dẫn đến mô hình học phân phối sai thực tế.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# ── ImageNet normalisation stats ─────────────────────────────────────────────
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def _make_train_transform(img_size: int) -> transforms.Compose:
    """
    Augmentation cho tập train (KHÔNG có HorizontalFlip – lý do y khoa).

    Các augmentation được dùng:
      - RandomResizedCrop   : mô phỏng cắt/zoom ngẫu nhiên, an toàn về giải phẫu
      - RandomRotation(±10°): biến động nhỏ góc chụp, giữ nguyên cấu trúc
      - ColorJitter         : mô phỏng khác biệt thiết bị/phơi sáng
      - Normalize           : chuẩn hoá ImageNet
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def _make_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def _compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Tính class weight nghịch đảo tần số để xử lý mất cân bằng nhãn."""
    counts = np.bincount(dataset.targets)
    weights = 1.0 / counts.astype(float)
    weights = weights / weights.sum() * len(counts)   # normalize
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size_train: int = 32,
    batch_size_eval: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, list, torch.Tensor]:
    """
    Tạo train / val / test DataLoader từ cấu trúc thư mục chuẩn:

        data_dir/
          train/  class_a/  ...
                  class_b/  ...
          val/    class_a/  ...
                  class_b/  ...
          test/   class_a/  ...
                  class_b/  ...

    Returns
    -------
    train_loader, val_loader, test_loader, class_names, class_weights
    """
    root = Path(data_dir)
    train_tf = _make_train_transform(img_size)
    eval_tf  = _make_eval_transform(img_size)

    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(root / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(root / "test",  transform=eval_tf)

    # Weighted sampler → cân bằng class khi sample batch
    class_weights = _compute_class_weights(train_ds)
    sample_weights = class_weights[train_ds.targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    print(f"[Data] Classes : {train_ds.classes}")
    print(f"[Data] Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")
    print(f"[Data] Class weights : {class_weights.tolist()}")

    return train_loader, val_loader, test_loader, train_ds.classes, class_weights