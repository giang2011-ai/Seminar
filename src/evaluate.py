"""
evaluate.py
-----------
ÄÃ¡nh giÃ¡ mÃ´ hÃ¬nh trÃªn táº­p test vá»›i cÃ¡c thang Ä‘o:
  - Accuracy, F1 (Macro & Weighted), Precision, Recall
  - AUC-ROC (OvR, macro)
  - Confusion Matrix
  - Classification Report Ä‘áº§y Ä‘á»§
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
    Cháº¡y inference trÃªn toÃ n bá»™ loader.

    Returns
    -------
    y_true   : shape (N,)        â€“ nhÃ£n thá»±c
    y_pred   : shape (N,)        â€“ nhÃ£n dá»± Ä‘oÃ¡n
    y_proba  : shape (N, C)      â€“ xÃ¡c suáº¥t softmax cho tá»«ng class
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
    ÄÃ¡nh giÃ¡ mÃ´ hÃ¬nh vÃ  in report toÃ n diá»‡n.

    Parameters
    ----------
    model : nn.Module
        Model Ä‘Ã£ Ä‘Æ°á»£c load best weights.
    test_loader : DataLoader
    class_names : list[str]
    device : torch.device
    output_dir : str
        ThÆ° má»¥c lÆ°u numpy arrays (Ä‘á»ƒ váº½ biá»ƒu Ä‘á»“ sau).
    model_name : str

    Returns
    -------
    metrics : dict
        Chá»©a táº¥t cáº£ chá»‰ sá»‘ Ä‘Ã¡nh giÃ¡.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(class_names)

    print(f"\n{'='*60}")
    print(f"  ÄÃ¡nh giÃ¡: {model_name.upper()} trÃªn táº­p TEST")
    print(f"{'='*60}")

    y_true, y_pred, y_proba = get_predictions(model, test_loader, device, n_classes)

    # â”€â”€ CÃ¡c chá»‰ sá»‘ cÆ¡ báº£n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    acc       = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weight = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_macro= precision_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_macro = recall_score(y_true, y_pred,    average="macro",    zero_division=0)

    # â”€â”€ AUC-ROC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        print("  âš  KhÃ´ng Ä‘á»§ dá»¯ liá»‡u Ä‘á»ƒ tÃ­nh AUC-ROC.")

    # â”€â”€ Confusion Matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cm = confusion_matrix(y_true, y_pred)

    # â”€â”€ Classification Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    # â”€â”€ In káº¿t quáº£ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n  Accuracy        : {acc:.4f}")
    print(f"  F1-Macro        : {f1_macro:.4f}")
    print(f"  F1-Weighted     : {f1_weight:.4f}")
    print(f"  Precision-Macro : {prec_macro:.4f}")
    print(f"  Recall-Macro    : {rec_macro:.4f}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{report}")

    # â”€â”€ LÆ°u arrays Ä‘á»ƒ váº½ ROC curve vÃ  Confusion Matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    metrics_by_model: dict[str, dict],
    metric_keys: list[str] | None = None,
) -> None:
    """Print a comparison table for two or more models."""
    if metric_keys is None:
        metric_keys = ["accuracy", "f1_macro", "f1_weighted", "recall_macro", "auc_roc"]

    labels = {
        "accuracy": "Accuracy",
        "f1_macro": "F1-Macro",
        "f1_weighted": "F1-Weighted",
        "precision_macro": "Precision-Macro",
        "recall_macro": "Recall-Macro",
        "auc_roc": "AUC-ROC",
    }

    model_names = list(metrics_by_model.keys())
    name_width = max(12, max(len(name) for name in model_names))
    table_width = 22 + (name_width + 2) * len(model_names)

    print(f"\n{'=' * table_width}")
    print("  MODEL COMPARISON")
    header = f"  {'Metric':<20}" + "".join(
        f"{name:>{name_width + 2}}" for name in model_names
    )
    print(header)
    print(f"  {'-' * (table_width - 4)}")

    for key in metric_keys:
        row_values = [metrics_by_model[name].get(key, float("nan")) for name in model_names]
        best_value = np.nanmax(row_values)
        row = f"  {labels.get(key, key):<20}"
        for value in row_values:
            marker = "*" if np.isfinite(value) and value == best_value else " "
            row += f"{value:>{name_width + 1}.4f}{marker}"
        print(row)

    print(f"{'=' * table_width}\n")
