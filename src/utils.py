# -*- coding: utf-8 -*-
"""
src/utils.py
------------
Hàm vẽ biểu đồ và tiện ích chung.

Export:
    set_seed(seed)
    plot_training_curves(history, model_name, output_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, model_name, output_dir)
    plot_roc_curve(y_true, y_proba, class_names, model_name, output_dir)
    plot_model_comparison(metrics_by_model, output_dir)
    plot_all(histories, metrics_by_model, class_names, output_dir)
    save_config(config, output_dir)
    print_hardware_info()
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# ============================================================================
# Seed
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """Cố định seed toàn cục để tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Đảm bảo deterministic (có thể chậm hơn một chút)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[Seed] Global seed set to {seed}")


# ============================================================================
# Hardware info
# ============================================================================

def print_hardware_info() -> None:
    """In thông tin GPU/CPU để đưa vào bảng báo cáo."""
    print("\n── Hardware Info ──────────────────────────────────")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU       : {props.name}")
        print(f"  VRAM total: {props.total_memory / 1e9:.1f} GB")
        print(f"  CUDA      : {torch.version.cuda}")
        print(f"  cuDNN     : {torch.backends.cudnn.version()}")
        # Bật TF32 để tăng tốc trên Ampere+ (RTX 30/40xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        print("  TF32      : ENABLED (Ampere+ speedup)")
    else:
        print("  GPU: Not available, using CPU")
    print(f"  PyTorch   : {torch.__version__}")
    print("──────────────────────────────────────────────────\n")


# ============================================================================
# Config save
# ============================================================================

def save_config(config: dict, output_dir: str = "./outputs") -> None:
    """Lưu CONFIG dict ra file JSON để tái lập thực nghiệm."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[Config] Saved to {path}")


# ============================================================================
# Training curves
# ============================================================================

def plot_training_curves(
    history: Dict[str, List],
    model_name: str = "model",
    output_dir: str = "./outputs",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_title(f"{model_name} – Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_f1"], label="Train F1")
    axes[1].plot(epochs, history["val_f1"],   label="Val F1")
    axes[1].set_title(f"{model_name} – F1 Score (Macro)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved {path}")
    return fig


# ============================================================================
# Confusion Matrix
# ============================================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: List[str],
    model_name: str = "model",
    output_dir: str = "./outputs",
) -> plt.Figure:
    from sklearn.metrics import confusion_matrix

    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalized

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, fmt in [
        (axes[0], cm,   "Raw counts",   "d"),
        (axes[1], cm_n, "Normalized",   ".2f"),
    ]:
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{model_name} – Confusion Matrix ({title})")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax)

        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = f"{data[i,j]:{fmt}}"
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved {path}")
    return fig


# ============================================================================
# ROC Curve
# ============================================================================

def plot_roc_curve(
    y_true,
    y_proba,
    class_names: List[str],
    model_name: str = "model",
    output_dir: str = "./outputs",
) -> plt.Figure:
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_classes))

    auc_scores = []
    for i, (cname, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], np.array(y_proba)[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cname} (AUC={roc_auc:.3f})")

    macro_auc = np.mean(auc_scores)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"{model_name} – ROC Curve (macro AUC={macro_auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved {path}")
    return fig


# ============================================================================
# Model comparison bar chart
# ============================================================================

def plot_model_comparison(
    metrics_by_model: Dict[str, Dict],
    output_dir: str = "./outputs",
) -> plt.Figure:
    model_names = list(metrics_by_model.keys())
    metrics_keys = ["accuracy", "f1", "precision", "recall", "auc"]
    labels       = ["Accuracy", "F1", "Precision", "Recall", "AUC"]

    x     = np.arange(len(model_names))
    width = 0.15
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (key, label, color) in enumerate(zip(metrics_keys, labels, colors)):
        vals = [metrics_by_model[m].get(key, 0) for m in model_names]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_title("Model Comparison – Test Metrics")
    ax.set_xticks(x + width * (len(metrics_keys) - 1) / 2)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved {path}")
    return fig


# ============================================================================
# plot_all convenience wrapper
# ============================================================================

def plot_all(
    histories: Dict[str, Dict],
    metrics_by_model: Dict[str, Dict],
    class_names: List[str],
    output_dir: str = "./outputs",
) -> None:
    """Vẽ tất cả biểu đồ trong một lần gọi."""
    for name, hist in histories.items():
        plot_training_curves(hist, model_name=name, output_dir=output_dir)

    for name, metrics in metrics_by_model.items():
        plot_confusion_matrix(
            metrics["y_true"], metrics["y_pred"],
            class_names, model_name=name, output_dir=output_dir,
        )
        plot_roc_curve(
            metrics["y_true"], metrics["y_proba"],
            class_names, model_name=name, output_dir=output_dir,
        )

    plot_model_comparison(metrics_by_model, output_dir=output_dir)
