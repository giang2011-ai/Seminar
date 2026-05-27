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
    │
    └─► + shortcut (identity hoặc projection nếu cần đổi dim)
        ReLU
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================================
# Bottleneck Block
# ============================================================================

class Bottleneck(nn.Module):
    """
    Bottleneck Block của ResNet-50/101/152.

    Dùng cấu trúc 1×1 → 3×3 → 1×1 thay vì 2 lớp 3×3 (như ResNet-18/34)
    để giảm số params trong khi giữ nguyên độ sâu biểu diễn.

    expansion = 4 : chiều ra = planes × 4

    Ví dụ Layer 1: planes=64  → in=64,  out=256
              Layer 2: planes=128 → in=128, out=512
              Layer 3: planes=256 → in=256, out=1024
              Layer 4: planes=512 → in=512, out=2048

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
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        # ── 1×1 Conv: giảm chiều (bottleneck narrow) ─────────────────────────
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

        # ── 1×1 Conv: tăng chiều trở lại (bottleneck wide) ───────────────────
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample   # projection shortcut (có thể None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x                        # giữ lại input cho skip connection

        # ── Main path ────────────────────────────────────────────────────────
        out = self.relu(self.bn1(self.conv1(x)))    # 1×1
        out = self.relu(self.bn2(self.conv2(out)))  # 3×3
        out = self.bn3(self.conv3(out))             # 1×1 (chưa ReLU)

        # ── Shortcut ─────────────────────────────────────────────────────────
        # Nếu kích thước thay đổi (stride hoặc số kênh) → dùng projection
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
    ResNet-50 tự cài đặt hoàn toàn theo paper gốc.

    Cấu hình ResNet-50:
        layers = [3, 4, 6, 3]   ← số Bottleneck block mỗi layer
        planes = [64, 128, 256, 512]

    Parameters
    ----------
    n_classes : int
        Số lớp phân loại đầu ra.
    dropout : float
        Dropout trước lớp Linear cuối.
    """

    # Số Bottleneck block tại mỗi layer (đặc trưng của ResNet-50)
    LAYERS = [3, 4, 6, 3]

    def __init__(
        self,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_classes   = n_classes
        self.in_channels = 64          # số kênh sau stem, tăng dần qua các layer

        # ── Stem ─────────────────────────────────────────────────────────────
        # 224×224 → 112×112
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # 112×112 → 56×56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── 4 Residual Layer ─────────────────────────────────────────────────
        self.layer1 = self._make_layer(planes=64,  num_blocks=self.LAYERS[0], stride=1)
        self.layer2 = self._make_layer(planes=128, num_blocks=self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(planes=256, num_blocks=self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(planes=512, num_blocks=self.LAYERS[3], stride=2)

        # ── Head ─────────────────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # Global Average Pooling
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(512 * Bottleneck.expansion, n_classes)

        # ── Weight initialization ─────────────────────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Tạo một residual layer gồm nhiều Bottleneck block.

        Block đầu tiên xử lý việc thay đổi stride và số kênh (nếu cần),
        các block còn lại dùng stride=1 và in_channels đã được cập nhật.

        Parameters
        ----------
        planes : int
            Số kênh bên trong block (trước expansion).
        num_blocks : int
            Số Bottleneck block trong layer này.
        stride : int
            Stride của block đầu tiên.
        """
        downsample = None
        out_channels = planes * Bottleneck.expansion

        # Cần projection shortcut khi:
        #   - stride != 1  (spatial size thay đổi), hoặc
        #   - in_channels != out_channels (số kênh thay đổi)
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        blocks: list[nn.Module] = []

        # Block đầu: có thể downsample spatial + thay đổi số kênh
        blocks.append(Bottleneck(self.in_channels, planes, stride, downsample))
        self.in_channels = out_channels   # cập nhật cho các block sau

        # Các block còn lại: stride=1, kích thước giữ nguyên
        for _ in range(1, num_blocks):
            blocks.append(Bottleneck(self.in_channels, planes))

        return nn.Sequential(*blocks)

    # ------------------------------------------------------------------
    def _init_weights(self):
        """
        Khởi tạo weights theo chuẩn paper:
          - Conv2d  : kaiming_normal (mode=fan_out, nonlinearity=relu)
          - BatchNorm : weight=1, bias=0
          - Linear  : normal(std=0.01)
        """
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

    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất feature vector trước head.

        x   : (B, 3, 224, 224)
        out : (B, 2048)
        """
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 64,  112, 112)
        x = self.maxpool(x)                       # (B, 64,   56,  56)

        # Residual layers
        x = self.layer1(x)                        # (B, 256,  56,  56)
        x = self.layer2(x)                        # (B, 512,  28,  28)
        x = self.layer3(x)                        # (B, 1024, 14,  14)
        x = self.layer4(x)                        # (B, 2048,  7,   7)

        # Global Average Pooling
        x = self.avgpool(x)                       # (B, 2048,  1,   1)
        x = torch.flatten(x, 1)                   # (B, 2048)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)              # (B, 2048)
        x = self.dropout(x)
        x = self.fc(x)                            # (B, n_classes)
        return x

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True