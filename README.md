<div align="center">
---

# PHÂN LOẠI MỨC ĐỘ HỨNG THÚ HỌC TẬP SINH VIÊN<br> BẰNG PHÂN TÍCH KHUÔN MẶT

**Môn học:** XỬ LÝ ẢNH SỐ  
**Mã LHP:** DIPR430685_04  
**GVHD:** PGS.TS. Hoàng Văn Dũng
TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT THÀNH PHỐ HỒ CHÍ MINH
KHOA CÔNG NGHỆ THÔNG TIN
---

### NHÓM THỰC HIỆN: NHÓM 16

| STT | Họ và tên | Mã số sinh viên |
|:---:|:---|:---:|
| 1 | Huỳnh Ngọc Thạch | 23133072 |
| 2 | Huỳnh Hữu Huy | 23133027 |
| 3 | Đỗ Kiến Hưng | 23133030 |
| 4 | Nguyễn Tân Thành | 23133068 |

</div>

---

## 📋 Mục lục
- [Giới thiệu](#-giới-thiệu)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Xử lý dữ liệu](#-xử-lý-dữ-liệu)
- [Chạy Demo](#-chạy-demo)

---

## 📖 Giới thiệu

Dự án này xây dựng một hệ thống đánh giá mức độ tập trung của sinh viên trong thời gian thực thông qua phân tích hình ảnh khuôn mặt. Hệ thống kết hợp các kỹ thuật Computer Vision và Machine Learning để đưa ra dự đoán chính xác.

**Quy trình xử lý:**
1.  **Face Detection**: Phát hiện khuôn mặt từ luồng video webcam.
2.  **Preprocessing**: Tiền xử lý ảnh (Grayscale, Gaussian Filter, Histogram Equalization).
3.  **Feature Extraction**: Trích xuất đặc trưng sử dụng mạng **CNN** tùy chỉnh và phương pháp **HOG**.
4.  **Classification**: Phân loại trạng thái (Tập trung / Không tập trung) sử dụng thuật toán **SVM**.

🔗 **Dữ liệu tham khảo**: [Kaggle Dataset](https://www.kaggle.com/code/sheimsaad/notebook-student-eng/input)

---

## 📂 Cấu trúc dự án

```text
student-engagement-system/
├── configs/            # File cấu hình (YAML)
├── data/               # Thư mục chứa dữ liệu
│   ├── raw/            # Dữ liệu thô (Input cho script xử lý)
│   └── processed/      # Dữ liệu đã xử lý (Output sau khi chạy script)
├── docs/               # Tài liệu dự án (Hướng dẫn sử dụng, UI Guide)
├── notebooks/          # Jupyter Notebooks (Training & Validating)
├── src/                # Mã nguồn chính
│   ├── data_processing/ # Scripts làm sạch và xử lý dữ liệu
│   ├── demo/           # Ứng dụng Demo UI (Kiến trúc Modular)
│   ├── face_detection/ # Module phát hiện khuôn mặt
│   └── visualization/  # Các hàm hiển thị kết quả
├── requirements.txt    # Danh sách thư viện phụ thuộc
└── README.md           # Tài liệu hướng dẫn này
```

---

## 💻 Yêu cầu hệ thống

-   **Hệ điều hành**: Windows 10/11, Linux (Ubuntu), hoặc macOS.
-   **Ngôn ngữ**: Python 3.8 trở lên.
-   **Phần cứng**: Webcam (để chạy demo thời gian thực).

---

## ⚙️ Cài đặt

Để thiết lập môi trường chạy dự án, vui lòng thực hiện theo các bước sau:

### 1. Khởi tạo môi trường ảo (Virtual Environment)
Việc sử dụng môi trường ảo giúp tránh xung đột phiên bản thư viện.

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
# hay
.venv\Scripts\python.exe
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Cài đặt thư viện
Cài đặt các gói thư viện cần thiết từ file `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🔄 Xử lý dữ liệu

Nếu bạn muốn chuẩn bị dữ liệu mới để huấn luyện lại mô hình, hãy làm theo quy trình chuẩn hóa dữ liệu dưới đây:

### Bước 1: Chuẩn bị dữ liệu thô
Sắp xếp dữ liệu ảnh vào thư mục `data/raw/Student-engagement-dataset/` theo cấu trúc:

```text
data/raw/Student-engagement-dataset/
├── Engaged/
│   ├── image_01.jpg
│   ├── image_02.png
│   └── ...
└── Not engaged/
    ├── image_01.jpg
    ├── image_02.png
    └── ...
```

### Bước 2: Chạy script xử lý chuẩn
Chạy script `dataset_cleaner.py` để tự động phát hiện khuôn mặt, cắt vùng ROI (Region of Interest), khử nhiễu và cân bằng sáng.

```bash
python src/data_processing/dataset_cleaner.py
```

### Bước 3: Kiểm tra kết quả
Dữ liệu đã qua xử lý sẽ được lưu tại: `data/processed/Student-engagement-dataset-clean/`.

---

## 🚀 Chạy Demo

Ứng dụng Demo cung cấp giao diện đồ họa (GUI) trực quan để kiểm thử mô hình theo thời gian thực.

### Khởi chạy ứng dụng

Từ thư mục gốc của dự án, chạy lệnh sau:

```bash
python src/demo/main.py
```

### Các chế độ xem (View Modes)

Ứng dụng hỗ trợ 3 chế độ xem:

1. **Chế độ Đơn (Single Mode)**:
   - Hiển thị 1 model tại một thời điểm
   - Chọn model: CNN+SVM, CNN Thuần, hoặc HOG+SVM
   - Hiển thị metrics chi tiết cho model được chọn

2. **Chế độ So sánh (Comparison Mode)**:
   - Hiển thị 2 models song song để so sánh
   - So sánh hiệu suất và kết quả prediction
   - Tính toán độ đồng thuận giữa 2 models

3. **Chế độ Thống kê (Stats Mode)**:
   - Hiển thị thống kê cho tất cả models
   - Thời gian xử lý của từng model
   - Tỷ lệ hứng thú tổng hợp

### Hướng dẫn sử dụng
1.  Đảm bảo Webcam đã được kết nối.
2.  Sau khi khởi chạy, giao diện sẽ hiển thị:
    *   **Bên trái**: Màn hình video với các mode selector và controls
    *   **Bên phải**: Bảng thống kê chi tiết (FPS, khuôn mặt, ánh sáng, metrics)
3.  Các tính năng:
    *   ✅ Chuyển đổi giữa 3 chế độ xem
    *   ✅ Điều chỉnh độ sáng camera
    *   ✅ Xem ảnh đã tiền xử lý
    *   ✅ Thống kê real-time
4.  Nhấn nút `X` trên thanh tiêu đề để tắt ứng dụng.

---
<div align="center">
    <i>Cảm ơn thầy và các bạn đã quan tâm theo dõi!</i>
</div>
