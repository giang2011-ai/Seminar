"""
dataset.py
----------
Tải dữ liệu ảnh X-quang và áp dụng augmentation phù hợp.
Hỗ trợ tính class weights cho dữ liệu mất cân bằng.
"""

import os
from collections import Counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Augmentation mạnh cho tập train nhằm giảm overfitting trên tập nhỏ
    và giúp mô hình tổng quát hơn với ảnh X-quang thực tế.
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.ToTensor(),
        # Chuẩn hoá theo ImageNet (pretrained backbone)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Không augmentation – chỉ resize + normalize cho val/test."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class XRayDataset(Dataset):
    """
    Dataset đọc ảnh X-quang từ cấu trúc thư mục:
        root/
        ├── class_A/
        │   ├── img1.jpg
        │   └── ...
        └── class_B/
            └── ...

    Parameters
    ----------
    root_dir : str
        Đường dẫn đến thư mục gốc (train / val / test).
    transform : callable, optional
        Các phép biến đổi ảnh.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Lấy danh sách class từ các thư mục con
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples: list[tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if self._is_image(fname):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Không tìm thấy ảnh nào trong '{root_dir}'. "
                "Kiểm tra lại cấu trúc thư mục."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_image(filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def get_labels(self) -> list[int]:
        """Trả về list nhãn (dùng để tính class weights)."""
        return [label for _, label in self.samples]

    # ------------------------------------------------------------------
    # PyTorch API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Đọc ảnh và chuyển sang RGB (đảm bảo 3 kênh dù ảnh grayscale)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: XRayDataset) -> torch.Tensor:
    """
    Tính class weights nghịch đảo tần suất để dùng với WeightedCrossEntropy.

    Công thức:
        w_i = N / (n_classes * count_i)

    Returns
    -------
    torch.Tensor of shape (n_classes,)
    """
    labels = dataset.get_labels()
    counter = Counter(labels)
    n_classes = len(dataset.classes)
    total = len(labels)

    weights = []
    for i in range(n_classes):
        count = counter.get(i, 1)  # tránh chia 0
        weights.append(total / (n_classes * count))

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"[Dataset] Class distribution : {dict(counter)}")
    print(f"[Dataset] Class weights       : {weight_tensor.tolist()}")
    return weight_tensor


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size_train: int = 32,
    batch_size_eval: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], torch.Tensor]:
    """
    Tạo DataLoader cho train / val / test.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names, class_weights
    """
    train_dataset = XRayDataset(
        root_dir=os.path.join(data_dir, "train"),
        transform=get_train_transforms(img_size),
    )
    val_dataset = XRayDataset(
        root_dir=os.path.join(data_dir, "val"),
        transform=get_val_transforms(img_size),
    )
    test_dataset = XRayDataset(
        root_dir=os.path.join(data_dir, "test"),
        transform=get_val_transforms(img_size),
    )

    # Đảm bảo 3 bộ có cùng tập class
    assert train_dataset.classes == val_dataset.classes, (
        "Train và Val có tập class khác nhau!"
    )

    class_weights = compute_class_weights(train_dataset)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,       # tránh batch cuối kích thước lẻ
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    print(
        f"[Dataset] Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)}"
    )
    print(f"[Dataset] Classes: {train_dataset.classes}")

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes,
        class_weights,
    )