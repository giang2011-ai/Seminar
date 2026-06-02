# So sánh CNN và Vision Transformer (ViT)

## Mục tiêu đề tài

Dự án này tìm hiểu và so sánh ba kiến trúc học sâu trong lĩnh vực thị giác máy tính:

- **ResNet-50**: mạng tích chập sâu với skip connection, cân bằng giữa độ chính xác và chi phí tính toán.
- **ConvNeXt-Tiny**: kiến trúc CNN hiện đại (2022) học theo thiết kế của Transformer nhưng vẫn dùng convolution.
- **ViT-Tiny/16 (Vision Transformer)**: chia ảnh thành patch và học quan hệ toàn cục qua self-attention.

Trọng tâm là so sánh CNN (ResNet-50, ConvNeXt-Tiny) với ViT về kiến trúc, yêu cầu dữ liệu, tốc độ, tài nguyên và độ chính xác trong điều kiện **train from scratch trên dataset nhỏ/vừa**.

## Thông tin thực hiện

- **Nhóm 11**
- **Thành viên:**
  - *Lê Thị Khánh Linh*
  - *Nguyễn Huyền Thương*
  - *Trần Bùi Hà Giang*

## Cấu trúc thư mục

```text
Seminar/
├── outputs/              # Checkpoint, biểu đồ, file đầu ra
├── src/
│   ├── __init__.py
│   ├── dataset.py        # build_dataloaders (không có HorizontalFlip)
│   ├── train.py          # train_model (lưu best + last checkpoint)
│   ├── evaluate.py       # evaluate_model, Sensitivity/Specificity
│   ├── utils.py          # set_seed, plot_*, save_config, print_hardware_info
│   └── models/
│       ├── __init__.py   # build_model factory
│       ├── resnet.py     # ResNet-50 tự cài đặt
│       ├── convnext.py   # ConvNeXt-Tiny tự cài đặt
│       └── vit.py        # ViT-Tiny/16 tự cài đặt
├── main.ipynb            # Notebook chính
├── README.md
└── requirements.txt
```

## Tổng quan 3 mô hình

### ResNet-50

Kiến trúc ResNet với Bottleneck block (1×1 → 3×3 → 1×1) và skip connection. Tự cài đặt theo paper gốc He et al. 2016. ~25M params.

### ConvNeXt-Tiny

CNN hiện đại (Liu et al. 2022) với Inverted Bottleneck, DWConv 7×7, LayerNorm, GELU và LayerScale. Hiệu quả hơn ResNet với kiến trúc đơn giản hơn. ~28M params.

### ViT-Tiny/16

Vision Transformer với patch_size=16, embed_dim=192, 12 lớp encoder, 3 heads. ~5.7M params nhưng yêu cầu nhiều dữ liệu hơn để học tốt.

## Bảng so sánh

| Tiêu chí | ResNet-50 | ConvNeXt-Tiny | ViT-Tiny/16 |
|---|---|---|---|
| Cơ chế | Bottleneck + Skip | Inverted Bottleneck + DWConv | Patch Embedding + Self-Attention |
| Params (approx) | ~25M | ~28M | ~5.7M |
| Inductive bias | Mạnh | Mạnh | Yếu |
| Nhu cầu dữ liệu | Nhỏ/vừa | Nhỏ/vừa | Lớn hoặc pre-trained |
| Receptive field | Cục bộ → toàn cục (theo chiều sâu) | Rộng hơn (DWConv 7×7) | Toàn cục ngay từ đầu |
| Tốc độ inference | Nhanh | Nhanh | Chậm hơn (attention O(N²)) |
| VRAM | Vừa | Vừa | Cao hơn |

## Lưu ý Y khoa – Không dùng RandomHorizontalFlip

**`src/dataset.py` không áp dụng `RandomHorizontalFlip`.**

Trong ảnh y tế (X-quang ngực, MRI, siêu âm tim), lật ngang làm thay đổi tính đối xứng giải phẫu (tim nằm trái, gan nằm phải). Ảnh lật ngang là một mẫu không tồn tại trong thực tế lâm sàng và khiến mô hình học phân phối sai. Augmentation an toàn được dùng: `RandomResizedCrop`, `RandomRotation(±10°)`, `ColorJitter`.

## Kết luận và Dự báo kết quả

Với điều kiện **train from scratch trên dataset nhỏ/vừa** (không có pre-trained weights):

- **ResNet-50** và **ConvNeXt-Tiny** được kỳ vọng vượt trội so với ViT nhờ **inductive bias mạnh** phù hợp với dữ liệu ảnh. Convolution tự nhiên khai thác tính cục bộ và tính dịch chuyển của ảnh, không cần học từ đầu như ViT.
- **ViT** có inductive bias yếu hơn — không có giả định gì về cấu trúc không gian. Để ViT đạt hiệu quả tương đương, cần **hàng chục nghìn ảnh** hoặc phải fine-tune từ checkpoint đã pre-train trên ImageNet-21k. Trong thực nghiệm này, ViT khả năng cao có accuracy và F1 thấp hơn 2 mô hình CNN.
- **ConvNeXt-Tiny** có thể nhỉnh hơn ResNet-50 nhờ thiết kế hiện đại hơn (DWConv 7×7, LayerScale, Stochastic Depth) và receptive field rộng hơn mà vẫn giữ được inductive bias của CNN.

**Lựa chọn thực tế:** Với bài toán y tế dataset hạn chế, ưu tiên ConvNeXt hoặc ResNet. ViT chỉ phát huy lợi thế khi có pre-trained model hoặc dataset lớn.

## Tiêu chí đánh giá

- **Accuracy**, **Precision**, **Recall**, **F1-score** (macro)
- **Sensitivity** (Độ nhạy) = TP / (TP + FN) — quan trọng trong y tế để không bỏ sót ca bệnh
- **Specificity** (Độ đặc hiệu) = TN / (TN + FP) — tránh chẩn đoán nhầm
- **Confusion Matrix**, **ROC-AUC**
- **Training time**, **Inference time** (ms/ảnh), **Model size** (params), **VRAM** (MB)

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách chạy

```bash
jupyter notebook main.ipynb
```

Chạy lần lượt các cell. Kết quả lưu vào `outputs/`.
