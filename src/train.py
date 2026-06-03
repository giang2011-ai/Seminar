# -*- coding: utf-8 -*-
"""
src/train.py
------------
Vòng lặp huấn luyện chung cho ResNet-50, ConvNeXt-Tiny và ViT.

Tính năng:
  - Lưu best_model (val F1 cao nhất) và last_model (mỗi epoch)
  - Early stopping theo patience
  - Trả về history dict để vẽ learning curve
  - Hiển thị thanh tiến trình (progress bar) bằng tqdm
  - Linear warmup + Cosine annealing scheduler
  - Gradient clipping để ổn định training
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,               # ViT: 1e-4 | ResNet/ConvNeXt: 3e-4
    weight_decay: float = 1e-2,     # Tăng từ 1e-4 → regularization mạnh hơn
    warmup_epochs: int = 3,         # Linear warmup để ổn định đầu training
    grad_clip: float = 1.0,         # Gradient clipping chống exploding gradient
    label_smoothing: float = 0.1,   # Giảm overconfidence → generalize tốt hơn
    patience: int = 7,
    output_dir: str = "./outputs",
    model_name: str = "model",
) -> Dict[str, List]:
    """
    Huấn luyện model và trả về history.

    Checkpoint được lưu:
      outputs/{model_name}_best.pth  → model có val F1 cao nhất
      outputs/{model_name}_last.pth  → model ở epoch cuối cùng đã chạy
                                       (phòng mất điện / interrupt)

    Scheduler:
      - warmup_epochs đầu: Linear warmup từ lr/10 → lr
      - Còn lại: CosineAnnealingLR xuống lr * 0.01

    Gợi ý lr theo model:
      - ViT-Small   : lr=1e-4  (nhạy cảm với lr cao)
      - ConvNeXt-T  : lr=3e-4
      - ResNet-50   : lr=3e-4
    """
    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, f"{model_name}_best.pth")
    last_path = os.path.join(output_dir, f"{model_name}_last.pth")

    model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,   # Tránh model overconfident → generalize tốt hơn
    )

    # Weight decay không áp dụng cho bias và LayerNorm (chuẩn cho ViT)
    decay_params     = [p for n, p in model.named_parameters() if "bias" not in n and "norm" not in n]
    no_decay_params  = [p for n, p in model.named_parameters() if "bias" in n or "norm" in n]
    optimizer = AdamW(
        [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    # Linear warmup → Cosine annealing
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=lr * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_f1":   [], "val_f1":   [],
        "epoch_time": [],
    }

    best_val_f1  = -1.0
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        all_pred, all_true = [], []

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs} [Train]", leave=False)

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping — quan trọng với ViT
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss += loss.item() * images.size(0)
            all_pred.extend(outputs.argmax(1).cpu().tolist())
            all_true.extend(labels.cpu().tolist())
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)
        train_f1    = f1_score(all_true, all_pred, average="macro", zero_division=0)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_pred, val_true = [], []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:3d}/{epochs} [Val]", leave=False)

            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_pred.extend(outputs.argmax(1).cpu().tolist())
                val_true.extend(labels.cpu().tolist())

        val_loss /= len(val_loader.dataset)
        val_f1    = f1_score(val_true, val_pred, average="macro", zero_division=0)
        elapsed   = time.time() - t0

        scheduler.step()

        # ── Logging ──────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["epoch_time"].append(elapsed)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{model_name}] Epoch {epoch:3d}/{epochs} | "
            f"Loss {train_loss:.4f}/{val_loss:.4f} | "
            f"F1 {train_f1:.4f}/{val_f1:.4f} | "
            f"LR {current_lr:.2e} | "
            f"Time {elapsed:.1f}s",
            flush=True
        )

        # ── Lưu last checkpoint ───────────────────────────────────────────────
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1,
            "history": history,
        }, last_path)

        # ── Lưu best checkpoint ───────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (val F1={best_val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"[{model_name}] Training done. Best val F1 = {best_val_f1:.4f}")
    return history