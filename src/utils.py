"""
utils.py
--------
Các hàm vẽ biểu đồ:
  1. Training curves (Loss & F1 theo epoch)
  2. Confusion Matrix (chuẩn hoá và raw)
  3. ROC Curve (per-class và macro-average)
  4. So sánh metrics hai mô hình (bar chart)
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Style chung
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

COLOR_RESNET = "#2196F3"   # Xanh dương
COLOR_VIT    = "#FF5722"   # Cam đỏ
COLOR_TRAIN  = "#4CAF50"   # Xanh lá
COLOR_VAL    = "#9C27B0"   # Tím


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: dict,
    model_name: str,
    output_dir: str = "outputs",
) -> None:
    """
    Vẽ biểu đồ Loss và F1-Macro theo epoch.

    Parameters
    ----------
    history : dict
        Kết quả trả về từ train_model().
    model_name : str
        Dùng làm tiêu đề và tên file.
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Curves – {model_name}", fontsize=14, fontweight="bold")

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=COLOR_TRAIN, label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   color=COLOR_VAL,   label="Val Loss",   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    # ── F1-Macro ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["train_f1"], color=COLOR_TRAIN, label="Train F1", linewidth=2)
    ax.plot(epochs, history["val_f1"],   color=COLOR_VAL,   label="Val F1",   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1-Macro")
    ax.set_title("F1-Macro")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Đã lưu: {save_path}")


# ---------------------------------------------------------------------------
# 2. Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_dir: str = "outputs",
    normalize: bool = True,
) -> None:
    """
    Vẽ Confusion Matrix (raw và chuẩn hoá theo hàng).
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    os.makedirs(output_dir, exist_ok=True)
    cm = sk_cm(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix – {model_name}", fontsize=14, fontweight="bold")

    for ax, norm, title in zip(
        axes,
        [False, True],
        ["Raw counts", "Normalized (row %)"],
    ):
        if norm:
            data = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
            vmax = 1.0
        else:
            data = cm
            fmt = "d"
            vmax = cm.max()

        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            linecolor="gray",
            vmin=0,
            vmax=vmax,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Đã lưu: {save_path}")


# ---------------------------------------------------------------------------
# 3. ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_dir: str = "outputs",
) -> None:
    """
    Vẽ ROC curve cho từng class (OvR) và macro-average.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_names)

    # Binarize labels
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Nếu binary thì y_bin shape (N,1); cần xử lý riêng
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    cmap = plt.cm.get_cmap("tab10", n_classes)
    fig, ax = plt.subplots(figsize=(8, 6))

    fpr_all = np.linspace(0, 1, 200)
    tpr_interp_list = []

    for i, cls_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        tpr_interp = np.interp(fpr_all, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp_list.append(tpr_interp)

        ax.plot(
            fpr, tpr,
            color=cmap(i),
            linewidth=1.8,
            label=f"{cls_name}  (AUC = {roc_auc:.3f})",
        )

    # Macro average
    mean_tpr = np.mean(tpr_interp_list, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = auc(fpr_all, mean_tpr)

    ax.plot(
        fpr_all, mean_tpr,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label=f"Macro-avg  (AUC = {macro_auc:.3f})",
    )

    # Đường chéo (random)
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle=":")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve – {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Đã lưu: {save_path}")


# ---------------------------------------------------------------------------
# 4. So sánh hai mô hình
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_resnet: dict,
    metrics_vit: dict,
    output_dir: str = "outputs",
) -> None:
    """
    Bar chart so sánh các chỉ số giữa ResNet-50 và ViT-B/16.
    """
    os.makedirs(output_dir, exist_ok=True)

    metric_keys   = ["accuracy", "f1_macro", "f1_weighted", "recall_macro", "auc_roc"]
    metric_labels = ["Accuracy", "F1-Macro", "F1-Weighted", "Recall-Macro", "AUC-ROC"]

    resnet_vals = [metrics_resnet.get(k, 0) for k in metric_keys]
    vit_vals    = [metrics_vit.get(k, 0)    for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, resnet_vals, width, label="ResNet-50", color=COLOR_RESNET, alpha=0.85)
    bars2 = ax.bar(x + width/2, vit_vals,    width, label="ViT-B/16",  color=COLOR_VIT,    alpha=0.85)

    # Ghi giá trị lên thanh bar
    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=COLOR_RESNET,
        )
    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=COLOR_VIT,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("So sánh ResNet-50 vs ViT-B/16 trên tập Test", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Đã lưu: {save_path}")


# ---------------------------------------------------------------------------
# 5. Vẽ tất cả biểu đồ trong một lần gọi
# ---------------------------------------------------------------------------

def plot_all(
    history: dict,
    metrics: dict,
    class_names: list[str],
    model_name: str,
    output_dir: str = "outputs",
) -> None:
    """Tiện ích gọi tất cả hàm vẽ biểu đồ cho một model."""
    plot_training_curves(history, model_name, output_dir)
    plot_confusion_matrix(
        metrics["y_true"], metrics["y_pred"],
        class_names, model_name, output_dir,
    )
    plot_roc_curve(
        metrics["y_true"], metrics["y_proba"],
        class_names, model_name, output_dir,
    )