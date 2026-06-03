"""
convnext.py
-----------
ConvNeXt-Tiny TỰ CÀI ĐẶT LẠI hoàn toàn từ đầu theo paper:
  "A ConvNet for the 2020s" – Liu et al., 2022
  https://arxiv.org/abs/2201.03545

Không dùng timm hay torchvision.models.convnext.

Kiến trúc tổng thể (ConvNeXt-Tiny):
  ┌──────────────────────────────────────────────────┐
  │  Patchify Stem : Conv2d(3→96, k=4, s=4)          │
  │                  + LayerNorm                      │
  ├──────────────────────────────────────────────────┤
  │  Stage 1 : 3 × ConvNeXtBlock  (C = 96 )          │
  │  Downsample   : LN + Conv2d(96→192, k=2, s=2)   │
  │  Stage 2 : 3 × ConvNeXtBlock  (C = 192)          │
  │  Downsample   : LN + Conv2d(192→384, k=2, s=2)  │
  │  Stage 3 : 9 × ConvNeXtBlock  (C = 384)          │
  │  Downsample   : LN + Conv2d(384→768, k=2, s=2)  │
  │  Stage 4 : 3 × ConvNeXtBlock  (C = 768)          │
  ├──────────────────────────────────────────────────┤
  │  Global Average Pooling  (B,768,H,W) → (B,768)   │
  │  LayerNorm → Dropout → Linear(768 → n_classes)   │
  └──────────────────────────────────────────────────┘

Mỗi ConvNeXtBlock (inverted bottleneck):
  input (B, C, H, W)
    │
    ├─► DWConv 7×7 (groups=C, padding=3)
    │   permute → (B,H,W,C)
    │   LayerNorm
    │   Linear C → 4C   [pointwise expand]
    │   GELU
    │   Linear 4C → C   [pointwise project]
    │   permute → (B,C,H,W)
    │   LayerScale (γ khởi tạo = 1e-6)
    │   StochasticDepth
    │
    └─► + residual
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================================
# Thành phần phụ trợ
# ============================================================================

class _ChannelLastLayerNorm(nn.Module):
    """
    Wrapper áp dụng nn.LayerNorm trên chiều channel của tensor
    channel-first (B, C, H, W).

    PyTorch LayerNorm kỳ vọng input dạng (..., C) nên cần permute trước/sau.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)   # (B, C, H, W) → (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # (B, H, W, C) → (B, C, H, W)
        return x


class LayerScale(nn.Module):
    """
    Learnable per-channel scalar γ ∈ ℝ^C, khởi tạo rất nhỏ (1e-6).

    Mục đích: ổn định gradient ở các layer sâu khi residual chưa được
    học đủ → tránh explosion/vanishing ở epoch đầu.

    Tham chiếu: Section 3.4 – ConvNeXt paper + DeiT III.
    """

    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        # nn.Parameter → được tối ưu cùng model
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape gamma (C,) → (1, C, 1, 1) để broadcast với (B, C, H, W)
        return x * self.gamma.view(1, -1, 1, 1)


class StochasticDepth(nn.Module):
    """
    Stochastic Depth (Drop Path) – regularization bằng cách bỏ ngẫu nhiên
    toàn bộ residual branch trong lúc training.

    Tham chiếu: "Deep Networks with Stochastic Depth" – Huang et al., 2016.

    Parameters
    ----------
    drop_prob : float
        Xác suất DROP một sample. Ở test time luôn = 0 (identity).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob

        # Mask shape (B, 1, 1, 1) → broadcast theo spatial + channel
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_prob)

        # Re-scale để giữ expectation không đổi (unbiased estimator)
        return x * mask / keep_prob


# ============================================================================
# ConvNeXt Block
# ============================================================================

class ConvNeXtBlock(nn.Module):
    """
    Đơn vị cơ bản của ConvNeXt, tự cài đặt theo Algorithm 1 trong paper.

    Điểm khác biệt so với ResNet Bottleneck Block:
      ┌────────────────┬────────────────────────────┬──────────────────────┐
      │                │ ResNet Bottleneck           │ ConvNeXt Block       │
      ├────────────────┼────────────────────────────┼──────────────────────┤
      │ Conv kernel    │ 1×1 → 3×3 → 1×1           │ 7×7 depthwise        │
      │ Normalisation  │ BatchNorm                  │ LayerNorm            │
      │ Activation     │ ReLU                       │ GELU                 │
      │ Bottleneck     │ Compress (C → C/4 → C)     │ Expand (C → 4C → C) │
      │ Regularisation │ Dropout (head only)         │ StochasticDepth      │
      └────────────────┴────────────────────────────┴──────────────────────┘

    Parameters
    ----------
    dim : int
        Số kênh đầu vào = đầu ra (không thay đổi trong block).
    expand_ratio : int
        Hệ số mở rộng của inverted bottleneck (mặc định 4 theo paper).
    drop_path_prob : float
        Xác suất stochastic depth.
    layer_scale_init : float
        Giá trị khởi tạo của γ trong LayerScale.
    """

    def __init__(
        self,
        dim: int,
        expand_ratio: int = 4,
        drop_path_prob: float = 0.0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = dim * expand_ratio

        # ── 1. Depthwise Conv 7×7 ────────────────────────────────────────────
        # groups=dim → mỗi kênh được conv bởi filter riêng (spatial mixing)
        self.dwconv = nn.Conv2d(
            dim, dim,
            kernel_size=7,
            padding=3,       # same padding: H, W không đổi
            groups=dim,
            bias=True,
        )

        # ── 2. LayerNorm (channel-last, PyTorch chuẩn) ───────────────────────
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # ── 3. Inverted Bottleneck (dùng Linear thay Conv 1×1) ───────────────
        # Lý do dùng Linear: sau khi permute về (B,H,W,C) thì Linear(C→4C)
        # tương đương Conv1×1 nhưng code rõ ràng hơn về ý định pointwise.
        self.pwconv1 = nn.Linear(dim, hidden_dim)    # expand  C → 4C
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)    # project 4C → C

        # ── 4. LayerScale + StochasticDepth ─────────────────────────────────
        self.layer_scale = LayerScale(dim, init_value=layer_scale_init)
        self.stoch_depth = StochasticDepth(drop_prob=drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)  →  out : (B, C, H, W)
        """
        residual = x                              # lưu lại cho skip-connection

        # --- Spatial mixing (depthwise, channel-first) ---
        x = self.dwconv(x)                        # (B, C, H, W)

        # --- Channel mixing (cần channel-last) ---
        x = x.permute(0, 2, 3, 1)                # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)                       # (B, H, W, 4C)
        x = self.act(x)
        x = self.pwconv2(x)                       # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)                # (B, C, H, W)

        # --- LayerScale + StochasticDepth + residual ---
        x = self.layer_scale(x)
        x = self.stoch_depth(x)
        return x + residual


# ============================================================================
# Downsampling giữa các stage
# ============================================================================

class ConvNeXtDownsample(nn.Module):
    """
    Giảm resolution 2× và tăng số kênh giữa hai stage liền kề.

    Cấu trúc: LayerNorm (channel-last) → Conv2d(k=2, s=2)

    Tại sao không dùng MaxPool?
    → Conv2d stride=2 học được cách downsample phù hợp với dữ liệu,
      trong khi MaxPool chỉ lấy max cố định.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)   # (B, C, H, W) → (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # (B, H, W, C) → (B, C, H, W)
        x = self.conv(x)             # (B, C', H/2, W/2)
        return x


# ============================================================================
# ConvNeXt-Tiny (full model)
# ============================================================================

class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny tự cài đặt hoàn toàn theo paper gốc.

    Cấu hình Tiny (Bảng 1 – paper):
        depths = [3, 3, 9, 3]       ← số ConvNeXtBlock mỗi stage
        dims   = [96, 192, 384, 768] ← số kênh mỗi stage

    Tổng ~28M params → nhẹ, phù hợp dataset y tế cỡ vừa.

    Parameters
    ----------
    n_classes : int
        Số lớp phân loại đầu ra.
    dropout : float
        Dropout trước lớp Linear cuối trong head.
    drop_path_rate : float
        Tổng stochastic depth rate. Mỗi block nhận giá trị tăng dần tuyến
        tính từ 0 đến drop_path_rate (block đầu ít regularize hơn block cuối).
    """

    DEPTHS = [3, 3, 9, 3]
    DIMS   = [96, 192, 384, 768]

    def __init__(
        self,
        n_classes: int = 2,
        dropout: float = 0.3,
        drop_path_rate: float = 0.2,   # Tăng từ 0.1 → regularize mạnh hơn
    ):
        super().__init__()
        self.n_classes = n_classes
        depths = self.DEPTHS
        dims   = self.DIMS

        # ── Patchify Stem ────────────────────────────────────────────────────
        # 224×224 → 56×56 (stride 4), 3 kênh RGB → 96 kênh
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            _ChannelLastLayerNorm(dims[0]),
        )

        # ── Phân bổ drop_path_prob tuyến tính qua tổng số block ─────────────
        total_blocks = sum(depths)
        dp_rates = [
            v.item()
            for v in torch.linspace(0.0, drop_path_rate, total_blocks)
        ]

        # ── 4 Stage + 3 Downsample xen kẽ ───────────────────────────────────
        self.stages      = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        block_idx = 0
        for i, (depth, dim) in enumerate(zip(depths, dims)):

            # Downsample trước stage i (stage 0 đã có stem, không cần thêm)
            if i == 0:
                self.downsamples.append(nn.Identity())   # placeholder stage 0
            else:
                self.downsamples.append(
                    ConvNeXtDownsample(dims[i - 1], dim)
                )

            # Xây dựng các ConvNeXtBlock của stage i
            blocks = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dim,
                    drop_path_prob=dp_rates[block_idx + j],
                )
                for j in range(depth)
            ])
            self.stages.append(blocks)
            block_idx += depth

        # ── Classification Head ──────────────────────────────────────────────
        self.head_norm    = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head_dropout = nn.Dropout(p=dropout)
        self.head_fc      = nn.Linear(dims[-1], n_classes)

        # ── Khởi tạo trọng số ───────────────────────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """
        Khởi tạo weights theo chuẩn ConvNeXt paper:
          - Conv2d / Linear : trunc_normal(std=0.02)
          - LayerNorm       : weight=1, bias=0
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất feature vector trước head.

        x   : (B, 3, 224, 224)
        out : (B, 768)
        """
        x = self.stem(x)                              # (B, 96,  56, 56)

        for i in range(len(self.stages)):
            x = self.downsamples[i](x)                # stage 0: identity
            x = self.stages[i](x)
        # Sau stage 4: (B, 768, 7, 7)

        # Global Average Pooling
        x = x.mean(dim=[-2, -1])                      # (B, 768)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)                  # (B, 768)
        x = self.head_norm(x)
        x = self.head_dropout(x)
        x = self.head_fc(x)                           # (B, n_classes)
        return x

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True