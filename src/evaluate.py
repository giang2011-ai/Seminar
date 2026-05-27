"""
evaluate.py
-----------
Đánh giá mô hình trên tập test với các thang đo:
  - Accuracy, F1 (Macro & Weighted), Precision, Recall
  - AUC-ROC (OvR, macro)
  - Confusion Matrix
  - Classification Report đầy đủ
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chạy inference trên toàn bộ loader.

    Returns
    -------
    y_true   : shape (N,)        – nhãn thực
    y_pred   : shape (N,)        – nhãn dự đoán
    y_proba  : shape (N, C)      – xác suất softmax cho từng class
    """
    model.eval()
    model = model.to(device)

    all_labels: list[int] = []
    all_preds:  list[int] = []
    all_proba:  list[np.ndarray] = []

    for images, labels in tqdm(loader, desc="  Inference", ncols=80, leave=False):
        images = images.to(device, non_blocking=True)

        logits = model(images)                          # (B, C)
        proba  = F.softmax(logits, dim=1).cpu().numpy() # (B, C)
        preds  = logits.argmax(dim=1).cpu().numpy()     # (B,)

        all_proba.append(proba)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_proba = np.vstack(all_proba)

    return y_true, y_pred, y_proba


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    output_dir: str = "outputs",
    model_name: str = "model",
) -> dict:
    """
    Đánh giá mô hình và in report toàn diện.

    Parameters
    ----------
    model : nn.Module
        Model đã được load best weights.
    test_loader : DataLoader
    class_names : list[str]
    device : torch.device
    output_dir : str
        Thư mục lưu numpy arrays (để vẽ biểu đồ sau).
    model_name : str

    Returns
    -------
    metrics : dict
        Chứa tất cả chỉ số đánh giá.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_names)

    print(f"\n{'='*60}")
    print(f"  Đánh giá: {model_name.upper()} trên tập TEST")
    print(f"{'='*60}")

    y_true, y_pred, y_proba = get_predictions(model, test_loader, device, n_classes)

    # ── Các chỉ số cơ bản ────────────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weight = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_macro= precision_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_macro = recall_score(y_true, y_pred,    average="macro",    zero_division=0)

    # ── AUC-ROC ───────────────────────────────────────────────────────────
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_true, y_proba,
                multi_class="ovr",
                average="macro",
            )
    except ValueError:
        auc = float("nan")
        print("  ⚠ Không đủ dữ liệu để tính AUC-ROC.")

    # ── Confusion Matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)

    # ── Classification Report ─────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    # ── In kết quả ───────────────────────────────────────────────────────
    print(f"\n  Accuracy        : {acc:.4f}")
    print(f"  F1-Macro        : {f1_macro:.4f}")
    print(f"  F1-Weighted     : {f1_weight:.4f}")
    print(f"  Precision-Macro : {prec_macro:.4f}")
    print(f"  Recall-Macro    : {rec_macro:.4f}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{report}")

    # ── Lưu arrays để vẽ ROC curve và Confusion Matrix ───────────────────
    np.save(os.path.join(output_dir, f"{model_name}_y_true.npy"), y_true)
    np.save(os.path.join(output_dir, f"{model_name}_y_pred.npy"), y_pred)
    np.save(os.path.join(output_dir, f"{model_name}_y_proba.npy"), y_proba)

    metrics = {
        "accuracy":         acc,
        "f1_macro":         f1_macro,
        "f1_weighted":      f1_weight,
        "precision_macro":  prec_macro,
        "recall_macro":     rec_macro,
        "auc_roc":          auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true":  y_true,
        "y_pred":  y_pred,
        "y_proba": y_proba,
    }

    return metrics


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_models(
    metrics_resnet: dict,
    metrics_vit: dict,
) -> None:
    """
    In bảng so sánh hai mô hình cạnh nhau.
    """
    keys = ["accuracy", "f1_macro", "f1_weighted", "recall_macro", "auc_roc"]
    labels = {
        "accuracy":     "Accuracy",
        "f1_macro":     "F1-Macro",
        "f1_weighted":  "F1-Weighted",
        "recall_macro": "Recall-Macro",
        "auc_roc":      "AUC-ROC",
    }

    print(f"\n{'='*60}")
    print("  SO SÁNH: ResNet-50 vs ViT-B/16")
    print(f"  {'Metric':<20} {'ResNet-50':>12} {'ViT-B/16':>12}")
    print(f"  {'-'*44}")
    for k in keys:
        r_val = metrics_resnet.get(k, float("nan"))
        v_val = metrics_vit.get(k, float("nan"))
        better = "↑" if v_val >= r_val else " "
        print(f"  {labels[k]:<20} {r_val:>12.4f} {v_val:>12.4f} {better}")
    print(f"{'='*60}\n")