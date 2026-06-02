# So sánh CNN và Vision Transformer (ViT)

## Mục tiêu đề tài

Dự án này được thực hiện nhằm tìm hiểu và so sánh hai kiến trúc học sâu phổ biến trong lĩnh vực thị giác máy tính:

- **CNN (Convolutional Neural Network)**: mạng nơ-ron tích chập, khai thác tốt đặc trưng cục bộ của ảnh thông qua các bộ lọc convolution.
- **ViT (Vision Transformer)**: mô hình Transformer cho ảnh, chia ảnh thành các patch và học quan hệ giữa các vùng ảnh bằng cơ chế self-attention.

Trọng tâm của đề tài là phân tích sự khác biệt giữa CNN và ViT về cách xử lý ảnh, kiến trúc mô hình, yêu cầu dữ liệu, ưu điểm, hạn chế và khả năng ứng dụng trong các bài toán phân loại ảnh.

## Thông tin thực hiện

- **Nhóm 11**
- **Thành viên:**
  - *Lê Thị Khánh Linh*
  - *Nguyễn Huyền Thương*
  - *Trần Bùi Hà Giang*

## Cấu trúc thư mục

```text
Seminar/
├── outputs/              # Lưu kết quả chạy mô hình, biểu đồ, ảnh hoặc file đầu ra
├── src/                  # Chứa mã nguồn hỗ trợ xử lý dữ liệu, mô hình và tiện ích
├── .gitattributes
├── .gitignore
├── main.ipynb            # Notebook chính để thực nghiệm và so sánh CNN với ViT
├── README.md             # Tài liệu mô tả đề tài
└── requirements.txt      # Danh sách thư viện cần cài đặt
```

## Tổng quan về CNN

CNN là kiến trúc mạng nơ-ron được thiết kế phù hợp với dữ liệu ảnh. Mô hình sử dụng các lớp tích chập để quét qua ảnh, từ đó học các đặc trưng như cạnh, góc, texture và các hình dạng phức tạp hơn ở những tầng sâu.

Đặc điểm chính của CNN:

- Khai thác tốt thông tin cục bộ giữa các pixel gần nhau.
- Có inductive bias mạnh đối với dữ liệu ảnh, đặc biệt là tính cục bộ và tính dịch chuyển.
- Thường hoạt động tốt trên tập dữ liệu nhỏ hoặc vừa.
- Có nhiều kiến trúc nổi tiếng như LeNet, AlexNet, VGG, ResNet, DenseNet và EfficientNet.

## Tổng quan về Vision Transformer

Vision Transformer đưa kiến trúc Transformer từ xử lý ngôn ngữ tự nhiên sang xử lý ảnh. Thay vì dùng các kernel tích chập, ViT chia ảnh thành nhiều patch nhỏ, biến mỗi patch thành một vector embedding, thêm thông tin vị trí và đưa vào Transformer Encoder.

Đặc điểm chính của ViT:

- Xem ảnh như một chuỗi các patch tương tự chuỗi token trong NLP.
- Dùng self-attention để học quan hệ toàn cục giữa các vùng ảnh.
- Có khả năng mở rộng tốt khi dữ liệu và tài nguyên tính toán đủ lớn.
- Thường đạt hiệu quả cao khi sử dụng mô hình đã được tiền huấn luyện.

## Bảng so sánh CNN và ViT

| Tiêu chí | CNN | ViT |
|---|---|---|
| Cách biểu diễn ảnh | Ảnh được xử lý trực tiếp qua các lớp tích chập | Ảnh được chia thành các patch rồi chuyển thành embedding |
| Cơ chế chính | Convolution, pooling, fully connected | Patch embedding, positional embedding, self-attention, MLP |
| Phạm vi học đặc trưng | Mạnh về đặc trưng cục bộ | Mạnh về quan hệ toàn cục giữa các patch |
| Inductive bias | Mạnh, phù hợp tự nhiên với ảnh | Yếu hơn, cần học nhiều hơn từ dữ liệu |
| Nhu cầu dữ liệu | Hiệu quả với dữ liệu nhỏ và vừa | Thường cần dữ liệu lớn hoặc pre-trained model |
| Chi phí tính toán | Thường nhẹ hơn ở các mô hình cơ bản | Có thể tốn bộ nhớ và thời gian hơn do self-attention |
| Khả năng mở rộng | Tốt, nhưng phụ thuộc nhiều vào thiết kế mạng | Rất tốt khi tăng kích thước mô hình và dữ liệu |
| Khả năng giải thích | Có thể quan sát feature map hoặc activation map | Có thể phân tích attention map |
| Ứng dụng phù hợp | Phân loại ảnh, nhận diện vật thể, y tế, hệ thống tài nguyên hạn chế | Bài toán dữ liệu lớn, mô hình hiện đại, bài toán cần học quan hệ toàn cục |

## Ưu điểm và hạn chế

### CNN

Ưu điểm:

- Phù hợp tự nhiên với dữ liệu ảnh.
- Dễ huấn luyện hơn khi dữ liệu không quá lớn.
- Chi phí tính toán thường hợp lý.
- Có nhiều mô hình nền tảng đã được kiểm chứng trong thực tế.

Hạn chế:

- Khả năng học quan hệ xa trong ảnh phụ thuộc vào độ sâu mạng và receptive field.
- Có thể bỏ sót thông tin toàn cục nếu kiến trúc chưa đủ phù hợp.
- Một số mô hình CNN sâu có nhiều tham số và cần kỹ thuật tối ưu tốt.

### ViT

Ưu điểm:

- Học được quan hệ toàn cục giữa các vùng ảnh thông qua self-attention.
- Có khả năng mở rộng tốt trên tập dữ liệu lớn.
- Linh hoạt, kế thừa nhiều thành tựu của Transformer.
- Hiệu quả cao khi fine-tune từ mô hình tiền huấn luyện.

Hạn chế:

- Cần nhiều dữ liệu nếu huấn luyện từ đầu.
- Chi phí tính toán và bộ nhớ có thể cao.
- Kém lợi thế hơn CNN trên dataset nhỏ nếu không có pre-training hoặc augmentation tốt.

## Quy trình thực nghiệm đề xuất

1. Chuẩn bị dữ liệu ảnh và tiền xử lý dữ liệu.
2. Xây dựng hoặc sử dụng mô hình CNN làm baseline.
3. Xây dựng hoặc fine-tune mô hình Vision Transformer.
4. Huấn luyện hai mô hình trong điều kiện thực nghiệm tương đương.
5. Đánh giá bằng các chỉ số như accuracy, precision, recall, F1-score và confusion matrix.
6. So sánh kết quả về độ chính xác, thời gian huấn luyện, thời gian suy luận và tài nguyên sử dụng.
7. Lưu kết quả, biểu đồ hoặc hình ảnh minh họa vào thư mục `outputs/`.

## Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Cách chạy

Mở notebook chính:

```bash
jupyter notebook main.ipynb
```

Sau đó chạy lần lượt các cell trong `main.ipynb` để thực hiện quá trình chuẩn bị dữ liệu, huấn luyện, đánh giá và so sánh mô hình CNN với ViT.

## Tiêu chí đánh giá

Các tiêu chí nên sử dụng khi so sánh hai mô hình:

- **Accuracy**: tỷ lệ dự đoán đúng trên toàn bộ tập kiểm tra.
- **Precision**: mức độ chính xác của các mẫu được dự đoán thuộc một lớp.
- **Recall**: khả năng tìm đúng các mẫu thực sự thuộc một lớp.
- **F1-score**: trung bình điều hòa giữa precision và recall.
- **Confusion matrix**: ma trận thể hiện số lượng dự đoán đúng và sai theo từng lớp.
- **Training time**: thời gian huấn luyện.
- **Inference time**: thời gian dự đoán.
- **Model size**: số lượng tham số hoặc dung lượng mô hình.

## Kết luận

CNN và ViT đều là những kiến trúc quan trọng trong thị giác máy tính. CNN có lợi thế khi dữ liệu hạn chế, chi phí tính toán cần tối ưu và bài toán yêu cầu mô hình ổn định, dễ triển khai. ViT có lợi thế khi có dữ liệu lớn hoặc sử dụng mô hình tiền huấn luyện, đặc biệt trong các bài toán cần học quan hệ toàn cục giữa nhiều vùng ảnh.

Việc lựa chọn CNN hay ViT phụ thuộc vào kích thước dữ liệu, tài nguyên tính toán, yêu cầu độ chính xác và mục tiêu triển khai thực tế của bài toán.
