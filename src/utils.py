"""
utils.py
--------
CÃ¡c hÃ m váº½ biá»ƒu Ä‘á»“:
  1. Training curves (Loss & F1 theo epoch)
  2. Confusion Matrix (chuáº©n hoÃ¡ vÃ  raw)
  3. ROC Curve (per-class vÃ  macro-average)
  4. So sÃ¡nh metrics hai mÃ´ hÃ¬nh (bar chart)
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

COLOR_RESNET = "#2196F3"   # Xanh dÆ°Æ¡ng
COLOR_VIT    = "#FF5722"   # Cam Ä‘á»
COLOR_TRAIN  = "#4CAF50"   # Xanh lÃ¡
COLOR_VAL    = "#9C27B0"   # TÃ­m


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: dict,
    model_name: str,
    output_dir: str = "outputs",
) -> None:
    """
    Váº½ biá»ƒu Ä‘á»“ Loss vÃ  F1-Macro theo epoch.

    Parameters
    ----------
    history : dict
        Káº¿t quáº£ tráº£ vá» tá»« train_model().
    model_name : str
        DÃ¹ng lÃ m tiÃªu Ä‘á» vÃ  tÃªn file.
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Curves â€“ {model_name}", fontsize=14, fontweight="bold")

    # â”€â”€ Loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=COLOR_TRAIN, label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   color=COLOR_VAL,   label="Val Loss",   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    # â”€â”€ F1-Macro â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    print(f"  [Plot] ÄÃ£ lÆ°u: {save_path}")


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
    Váº½ Confusion Matrix (raw vÃ  chuáº©n hoÃ¡ theo hÃ ng).
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    os.makedirs(output_dir, exist_ok=True)
    cm = sk_cm(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix â€“ {model_name}", fontsize=14, fontweight="bold")

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
    print(f"  [Plot] ÄÃ£ lÆ°u: {save_path}")


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
    Váº½ ROC curve cho tá»«ng class (OvR) vÃ  macro-average.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_names)

    # Binarize labels
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Náº¿u binary thÃ¬ y_bin shape (N,1); cáº§n xá»­ lÃ½ riÃªng
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

    # ÄÆ°á»ng chÃ©o (random)
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle=":")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve â€“ {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] ÄÃ£ lÆ°u: {save_path}")


# ---------------------------------------------------------------------------
# 4. So sÃ¡nh hai mÃ´ hÃ¬nh
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_by_model: dict[str, dict],
    output_dir: str = "outputs",
) -> None:
    """Bar chart comparing metrics for two or more models."""
    os.makedirs(output_dir, exist_ok=True)

    metric_keys = ["accuracy", "f1_macro", "f1_weighted", "recall_macro", "auc_roc"]
    metric_labels = ["Accuracy", "F1-Macro", "F1-Weighted", "Recall-Macro", "AUC-ROC"]
    model_names = list(metrics_by_model.keys())

    x = np.arange(len(metric_keys))
    width = min(0.8 / max(len(model_names), 1), 0.25)
    offsets = (np.arange(len(model_names)) - (len(model_names) - 1) / 2) * width
    colors = plt.cm.get_cmap("tab10", len(model_names))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model_name in enumerate(model_names):
        vals = [metrics_by_model[model_name].get(k, 0) for k in metric_keys]
        bars = ax.bar(
            x + offsets[i],
            vals,
            width,
            label=model_name,
            color=colors(i),
            alpha=0.88,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if len(model_names) > 2 else 0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model comparison on test set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved: {save_path}")

# ---------------------------------------------------------------------------
# 5. Váº½ táº¥t cáº£ biá»ƒu Ä‘á»“ trong má»™t láº§n gá»i
# ---------------------------------------------------------------------------

def plot_all(
    history: dict,
    metrics: dict,
    class_names: list[str],
    model_name: str,
    output_dir: str = "outputs",
) -> None:
    """Tiá»‡n Ã­ch gá»i táº¥t cáº£ hÃ m váº½ biá»ƒu Ä‘á»“ cho má»™t model."""
    plot_training_curves(history, model_name, output_dir)
    plot_confusion_matrix(
        metrics["y_true"], metrics["y_pred"],
        class_names, model_name, output_dir,
    )
    plot_roc_curve(
        metrics["y_true"], metrics["y_proba"],
        class_names, model_name, output_dir,
    )
