# -*- coding: utf-8 -*-
"""
src/evaluate.py
---------------
Đánh giá mô hình trên tập test.

Metrics:
  - Accuracy, Precision, Recall, F1-score (macro)
  - Sensitivity (= Recall per class)
  - Specificity per class  (TP của class khác / tổng negative thực sự)
  - Confusion Matrix
  - ROC-AUC (macro)
  - Model size (params)
  - VRAM thực tế (nếu CUDA)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================================
# Medical metrics: Sensitivity & Specificity
# ============================================================================

def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Tính Sensitivity và Specificity cho từng class theo OvR (One vs Rest).

    Sensitivity (Độ nhạy) = TP / (TP + FN)
        → Khả năng phát hiện đúng ca dương tính (quan trọng trong y tế
          để không bỏ sót bệnh).

    Specificity (Độ đặc hiệu) = TN / (TN + FP)
        → Khả năng xác nhận đúng ca âm tính (tránh chẩn đoán nhầm).

    Returns dict: { class_name: { "sensitivity": float, "specificity": float } }
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    results = {}

    for i, cname in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp          # false negatives
        fp = cm[:, i].sum() - tp          # false positives
        tn = cm.sum() - tp - fn - fp      # true negatives

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results[cname] = {
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
        }

    return results


# ============================================================================
# Model size & VRAM helpers
# ============================================================================

def get_model_info(model: nn.Module) -> Dict[str, int]:
    """Trả về số params tổng và trainable."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def get_vram_mb() -> float:
    """VRAM thực tế đang dùng (MB). Trả về 0 nếu không có CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


# ============================================================================
# Main evaluate function
# ============================================================================

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    output_dir: str = "./outputs",
    model_name: str = "model",
) -> Dict:
    """
    Chạy inference trên test set và tính đầy đủ metrics.

    Returns
    -------
    dict với keys:
        y_true, y_pred, y_proba,
        accuracy, f1, precision, recall,
        sensitivity_specificity,
        report, model_info, inference_time_ms, vram_mb
    """
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    model.eval()

    y_true, y_pred, y_proba = [], [], []

    # Reset VRAM peak counter trước khi đo
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_proba.extend(probs.cpu().tolist())

    inference_time_ms = (time.time() - t_start) * 1000 / len(test_loader.dataset)

    # VRAM peak trong suốt inference
    vram_mb = torch.cuda.max_memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0.0

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_proba = np.array(y_proba)

    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    sens_spec = compute_sensitivity_specificity(y_true, y_pred, class_names)
    model_info = get_model_info(model)

    # ── In kết quả ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {model_name.upper()} — Test Results")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Per-class Sensitivity & Specificity:")
    for cname, vals in sens_spec.items():
        print(f"    {cname:20s}  Sens={vals['sensitivity']:.4f}  Spec={vals['specificity']:.4f}")
    print(f"\n  Model params  : {model_info['total_params']:,}")
    print(f"  VRAM peak     : {vram_mb:.1f} MB")
    print(f"  Inference     : {inference_time_ms:.2f} ms/image")
    print(f"\n{report}")

    return {
        "y_true":   y_true,
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "accuracy": acc,
        "f1":       f1,
        "precision": precision,
        "recall":   recall,
        "auc":      auc,
        "sensitivity_specificity": sens_spec,
        "report":   report,
        "model_info": model_info,
        "inference_time_ms": inference_time_ms,
        "vram_mb":  vram_mb,
    }


def compare_models(metrics_by_model: Dict[str, Dict]) -> None:
    """In bảng so sánh tổng hợp các model."""
    print(f"\n{'='*80}")
    print("  MODEL COMPARISON")
    print(f"{'='*80}")
    header = f"  {'Model':<18} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Params':>12} {'VRAM(MB)':>10} {'ms/img':>8}"
    print(header)
    print("  " + "-" * 76)

    for name, m in metrics_by_model.items():
        params = m["model_info"]["total_params"]
        print(
            f"  {name:<18} "
            f"{m['accuracy']:>7.4f} "
            f"{m['f1']:>7.4f} "
            f"{m['auc']:>7.4f} "
            f"{params:>12,} "
            f"{m['vram_mb']:>10.1f} "
            f"{m['inference_time_ms']:>8.2f}"
        )
    print(f"{'='*80}\n")
