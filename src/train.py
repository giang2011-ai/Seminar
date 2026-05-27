"""
train.py
--------
Training loop với:
  - Weighted Cross Entropy Loss (xử lý mất cân bằng class)
  - Cosine Annealing LR Scheduler
  - Early Stopping
  - Lưu best model theo val F1-Macro
"""

from __future__ import annotations

import os
import time
import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Dừng training sớm nếu metric không cải thiện sau `patience` epoch.

    Parameters
    ----------
    patience : int
        Số epoch chờ trước khi dừng.
    min_delta : float
        Ngưỡng cải thiện tối thiểu.
    mode : str
        'max' nếu metric càng cao càng tốt (F1, AUC), 'min' nếu ngược lại (loss).
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Returns True nếu nên dừng training.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,          # torch.cuda.amp.GradScaler hoặc None
) -> tuple[float, float]:
    """
    Chạy một epoch train.

    Returns
    -------
    avg_loss : float
    f1_macro : float
    """
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in tqdm(loader, desc="  Train", leave=False, ncols=80):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed Precision (AMP)
            with torch.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Chạy một epoch evaluation (val hoặc test).

    Returns
    -------
    avg_loss : float
    f1_macro : float
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 7,
    output_dir: str = "outputs",
    model_name: str = "model",
) -> dict:
    """
    Huấn luyện mô hình với Weighted Cross Entropy + Cosine Annealing.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    class_weights : torch.Tensor
        Weights cho từng class, shape (n_classes,).
    device : torch.device
    epochs : int
    lr : float
        Learning rate ban đầu.
    weight_decay : float
    patience : int
        Patience cho EarlyStopping.
    output_dir : str
        Thư mục lưu model checkpoint.
    model_name : str
        Prefix tên file checkpoint.

    Returns
    -------
    history : dict
        Chứa list loss/f1 qua từng epoch để vẽ biểu đồ.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1,   # Label smoothing giảm overfit nhẹ
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── AMP scaler (chỉ khi có CUDA) ──────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Early stopping ────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=patience, mode="max")
    best_val_f1 = -1.0
    best_weights = None

    # ── History ───────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_f1": [],
        "val_loss":   [], "val_f1":   [],
        "lr":         [],
    }

    checkpoint_path = os.path.join(output_dir, f"{model_name}_best.pth")

    print(f"\n{'='*60}")
    print(f"  Bắt đầu training: {model_name.upper()}")
    print(f"  Epochs={epochs} | LR={lr} | Device={device}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_f1 = evaluate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        elapsed = time.time() - t0

        # Lưu history
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:>3}/{epochs}] "
            f"| Train Loss: {train_loss:.4f}  F1: {train_f1:.4f} "
            f"| Val Loss: {val_loss:.4f}  F1: {val_f1:.4f} "
            f"| LR: {current_lr:.2e} "
            f"| {elapsed:.1f}s"
        )

        # Lưu best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_weights,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved best model  (Val F1-Macro: {val_f1:.4f})")

        # Early stopping
        if early_stop.step(val_f1):
            print(f"\n  ⚠ Early stopping sau {epoch} epoch (patience={patience})")
            break

    print(f"\n{'='*60}")
    print(f"  Training hoàn tất. Best Val F1-Macro: {best_val_f1:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    # Nạp lại best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return history