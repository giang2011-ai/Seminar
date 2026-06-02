# src/models/__init__.py  (re-export từ __init__.py gốc ở root)
from src.models.convnext import ConvNeXtTiny
from src.models.resnet   import ResNet50Classifier
from src.models.vit      import VisionTransformerClassifier

__all__ = ["ConvNeXtTiny", "ResNet50Classifier", "VisionTransformerClassifier", "build_model"]

import torch.nn as nn

SUPPORTED_MODELS = {"resnet50", "convnext", "vit"}


def build_model(
    model_name: str,
    n_classes: int,
    img_size: int = 224,
    dropout: float = 0.3,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    name = model_name.lower()

    if name == "resnet50":
        model = ResNet50Classifier(n_classes=n_classes, dropout=dropout)
        print(f"[Model] ResNet-50 | n_classes={n_classes}")

    elif name == "convnext":
        model = ConvNeXtTiny(
            n_classes=n_classes,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )
        print(f"[Model] ConvNeXt-Tiny | n_classes={n_classes} | drop_path={drop_path_rate}")

    elif name == "vit":
        model = VisionTransformerClassifier(
            n_classes=n_classes,
            img_size=img_size,
            dropout=dropout,
        )
        print(f"[Model] ViT-Tiny/16 | n_classes={n_classes} | img_size={img_size}")

    else:
        raise ValueError(f"model_name must be one of {SUPPORTED_MODELS}, got: '{model_name}'")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total:,} | Trainable: {trainable:,}")

    return model
