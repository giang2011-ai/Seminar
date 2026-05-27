"""
src/models/__init__.py
----------------------
Factory function – gom ResNet-50 và ConvNeXt-Tiny.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.resnet   import ResNet50Classifier
from src.models.convnext import ConvNeXtTiny

SUPPORTED_MODELS = {"resnet50", "convnext"}


def build_model(
    model_name: str,
    n_classes: int,
    dropout: float = 0.3,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    name = model_name.lower()

    if name == "resnet50":
        model = ResNet50Classifier(
            n_classes=n_classes,
            dropout=dropout,
        )
        print(f"[Model] ResNet-50 (tự cài đặt) | n_classes={n_classes}")

    elif name == "convnext":
        model = ConvNeXtTiny(
            n_classes=n_classes,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        print(f"[Model] ConvNeXt-Tiny (tự cài đặt) | n_classes={n_classes} | drop_path={drop_path_rate}")

    else:
        raise ValueError(
            f"model_name phải thuộc {SUPPORTED_MODELS}, nhận được: '{model_name}'"
        )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Tổng params: {total:,} | Trainable: {trainable:,}")

    return model
