"""
resnet.py
---------
ResNet-50 TỰ CÀI ĐẶT LẠI hoàn toàn từ đầu theo paper gốc:
  "Deep Residual Learning for Image Recognition" – He et al., 2016
  https://arxiv.org/abs/1512.03385

Không dùng torchvision.models.resnet50.

Kiến trúc ResNet-50:
  ┌─────────────────────────────────────────────────────┐
  │  Conv 7×7, stride=2, 64 filters  (112×112)          │
  │  BatchNorm + ReLU                                   │
  │  MaxPool 3×3, stride=2           (56×56)            │
  ├─────────────────────────────────────────────────────┤
  │  Layer 1 : 3 × Bottleneck  (C=64,  out=256 )       │
  │  Layer 2 : 4 × Bottleneck  (C=128, out=512 )       │
  │  Layer 3 : 6 × Bottleneck  (C=256, out=1024)       │
  │  Layer 4 : 3 × Bottleneck  (C=512, out=2048)       │
  ├─────────────────────────────────────────────────────┤
  │  Global Average Pooling  → (B, 2048)                │
  │  Dropout → Linear(2048 → n_classes)                 │
  └─────────────────────────────────────────────────────┘

Mỗi Bottleneck Block:
  input (B, C_in, H, W)
    │
    ├─► Conv 1×1  (giảm chiều: C_in → planes)
    │   BatchNorm + ReLU
    │   Conv 3×3  (spatial feature extraction)
    │   BatchNorm + ReLU
    │   Conv 1×1  (tăng chiều: planes → planes×4)
    │   BatchNorm
    │   LayerScale (γ khởi tạo = 1e-6, ổn định gradient)
    │   DropPath  (Stochastic Depth, regularize sâu)
    │
    └─► + shortcut (identity hoặc projection nếu cần đổi dim)
        ReLU

Regularization bổ sung so với paper gốc:
  - LayerScale  : ổn định gradient ở layer sâu (như ConvNeXt)
  - DropPath    : Stochastic Depth tăng dần từng block (0 → drop_path_rate)
  - Dropout 0.3 : tại head trước fc
  Khuyến nghị dùng label_smoothing=0.1 trong CrossEntropyLoss ở train.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================================
# Thành phần phụ trợ
# ============================================================================

class DropPath(nn.Module):
    """
    Stochastic Depth – drop toàn bộ residual branch theo từng sample.
    Tương tự StochasticDepth trong convnext.py.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_prob)
        return x * mask / keep_prob


class LayerScale(nn.Module):
    """
    Learnable per-channel scalar γ, khởi tạo rất nhỏ (1e-6).
    Giúp ổn định gradient khi residual branch chưa được học đủ.
    Reshape γ (C,) → (1, C, 1, 1) để broadcast với (B, C, H, W).
    """

    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.view(1, -1, 1, 1)


# ============================================================================
# Bottleneck Block
# ============================================================================

class Bottleneck(nn.Module):
    """
    Bottleneck Block của ResNet-50/101/152 với Stochastic Depth và LayerScale.

    expansion = 4 : chiều ra = planes × 4

    Parameters
    ----------
    in_channels : int
        Số kênh đầu vào.
    planes : int
        Số kênh của 2 lớp conv bên trong (trước khi expand).
    stride : int
        Stride của conv 3×3. stride=2 → downsample spatial.
    downsample : nn.Module | None
        Projection shortcut khi in_channels ≠ planes×expansion.
    drop_path : float
        Stochastic Depth rate riêng cho block này.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # ── 1×1 Conv: giảm chiều ─────────────────────────────────────────────
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # ── 3×3 Conv: trích xuất đặc trưng không gian ────────────────────────
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # ── 1×1 Conv: tăng chiều trở lại ─────────────────────────────────────
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

        # ── Regularization: LayerScale + DropPath ────────────────────────────
        self.layer_scale = LayerScale(planes * self.expansion)
        self.drop_path   = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # ── Main path ────────────────────────────────────────────────────────
        out = self.relu(self.bn1(self.conv1(x)))    # 1×1
        out = self.relu(self.bn2(self.conv2(out)))  # 3×3
        out = self.bn3(self.conv3(out))             # 1×1 (chưa ReLU)

        # ── LayerScale + DropPath trước residual add ─────────────────────────
        out = self.layer_scale(out)
        out = self.drop_path(out)

        # ── Shortcut ─────────────────────────────────────────────────────────
        if self.downsample is not None:
            identity = self.downsample(x)

        # ── Residual add + ReLU ───────────────────────────────────────────────
        out = self.relu(out + identity)
        return out


# ============================================================================
# ResNet-50
# ============================================================================

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 tự cài đặt hoàn toàn theo paper gốc + regularization hiện đại.

    Cấu hình ResNet-50:
        layers = [3, 4, 6, 3]   ← số Bottleneck block mỗi layer
        planes = [64, 128, 256, 512]

    Parameters
    ----------
    n_classes : int
        Số lớp phân loại đầu ra.
    dropout : float
        Dropout trước lớp Linear cuối.
    drop_path_rate : float
        Stochastic Depth rate tối đa (block đầu = 0, block cuối = rate này).
        Mặc định 0.2 phù hợp với dataset y tế cỡ vừa.
    """

    LAYERS = [3, 4, 6, 3]

    def __init__(
        self,
        n_classes: int = 2,
        dropout: float = 0.3,
        drop_path_rate: float = 0.2,   # Stochastic Depth tăng dần
    ):
        super().__init__()
        self.n_classes   = n_classes
        self.in_channels = 64

        # Phân bổ drop_path_rate tuyến tính qua tổng số block (16 blocks)
        total_blocks = sum(self.LAYERS)
        self._dpr = [
            v.item()
            for v in torch.linspace(0.0, drop_path_rate, total_blocks)
        ]
        self._block_idx = 0   # con trỏ dùng trong _make_layer

        # ── Stem ─────────────────────────────────────────────────────────────
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── 4 Residual Layer ─────────────────────────────────────────────────
        self.layer1 = self._make_layer(planes=64,  num_blocks=self.LAYERS[0], stride=1)
        self.layer2 = self._make_layer(planes=128, num_blocks=self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(planes=256, num_blocks=self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(planes=512, num_blocks=self.LAYERS[3], stride=2)

        # ── Head ─────────────────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(512 * Bottleneck.expansion, n_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample   = None
        out_channels = planes * Bottleneck.expansion

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        blocks: list[nn.Module] = []
        for i in range(num_blocks):
            blocks.append(Bottleneck(
                in_channels=self.in_channels if i == 0 else out_channels,
                planes=planes,
                stride=stride if i == 0 else 1,
                downsample=downsample if i == 0 else None,
                drop_path=self._dpr[self._block_idx],   # rate tăng dần
            ))
            self._block_idx += 1

        self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 64,  112, 112)
        x = self.maxpool(x)                       # (B, 64,   56,  56)
        x = self.layer1(x)                        # (B, 256,  56,  56)
        x = self.layer2(x)                        # (B, 512,  28,  28)
        x = self.layer3(x)                        # (B, 1024, 14,  14)
        x = self.layer4(x)                        # (B, 2048,  7,   7)
        x = self.avgpool(x)                       # (B, 2048,  1,   1)
        x = torch.flatten(x, 1)                   # (B, 2048)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True