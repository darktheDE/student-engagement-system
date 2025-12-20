# Student Engagement Analysis System

Hệ thống đánh giá mức độ tập trung của sinh viên trong lớp học thông qua phân tích đặc trưng khuôn mặt sử dụng Computer Vision và Machine Learning.

## 📋 Mục lục
- [Giới thiệu](#-giới-thiệu)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Xử lý dữ liệu](#-xử-lý-dữ-liệu)
- [Chạy Demo](#-chạy-demo)

## 📖 Giới thiệu

Dự án này sử dụng Deep Learning (CNN) để trích xuất đặc trưng và Machine Learning (SVM) để phân loại trạng thái tập trung của sinh viên.
Hệ thống bao gồm các bước:
1.  **Face Detection**: Phát hiện khuôn mặt từ luồng video.
2.  **Preprocessing**: Xử lý ảnh (Grayscale, Filter, Histogram Equalization).
3.  **Feature Extraction**: Trích xuất đặc trưng bằng mô hình CNN.
4.  **Classification**: Phân loại trạng thái (Tập trung / Không tập trung) bằng SVM.

## 📂 Cấu trúc dự án

```
student-engagement-system/
├── configs/            # File cấu hình (YAML)
├── data/               # Thư mục chứa dữ liệu
│   ├── raw/            # Dữ liệu thô (Input cho script xử lý)
│   └── processed/      # Dữ liệu đã xử lý (Output sau khi chạy script)
├── docs/               # Tài liệu dự án
├── models/             # Chứa model đã train (CNN, SVM)
├── notebooks/          # Jupyter Notebooks cho quá trình train/test
├── src/                # Mã nguồn chính
│   ├── data_processing/ # Scripts xử lý dữ liệu
│   ├── demo/           # Ứng dụng Demo UI
│   ├── face_detection/ # Module phát hiện khuôn mặt
│   └── visualization/  # Các hàm hiển thị
├── requirements.txt    # Danh sách thư viện cần thiết
└── README.md           # Tài liệu hướng dẫn
```

## 💻 Yêu cầu hệ thống

-   **Python**: 3.8 hoặc mới hơn.
-   **Webcam**: Để chạy ứng dụng demo thời gian thực.
-   **OS**: Windows, Linux, hoặc macOS.

## ⚙️ Cài đặt

Để thiết lập môi trường chạy dự án, hãy làm theo các bước sau:

### 1. Tạo môi trường ảo (Virtual Environment)
Khuyến khích sử dụng môi trường ảo để tránh xung đột thư viện.

**Trên Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Trên Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Cài đặt các thư viện phụ thuộc
Sau khi kích hoạt môi trường ảo, chạy lệnh:

```bash
pip install -r requirements.txt
```

---

## 🔄 Xử lý dữ liệu

Nếu bạn muốn chuẩn bị dữ liệu mới để training, hãy làm theo quy trình sau:

1.  **Chuẩn bị dữ liệu thô**:
    Đặt dữ liệu ảnh của bạn vào thư mục `data/raw/Student-engagement-dataset/`.
    Cấu trúc thư mục mong đợi:
    ```
    data/raw/Student-engagement-dataset/
    ├── Engaged/
    │   ├── ... (các ảnh jpg, png)
    └── Not engaged/
        ├── ... (các ảnh jpg, png)
    ```

2.  **Chạy script xử lý**:
    Script này sẽ thực hiện phát hiện khuôn mặt, cắt vùng mặt, xử lý nhiễu và lưu vào thư mục `data/processed`.

    ```bash
    python src/data_processing/dataset_cleaner.py
    ```

3.  **Kết quả**:
    Dữ liệu đã xử lý sẽ nằm tại `data/processed/Student-engagement-dataset-clean/`.

---

## 🚀 Chạy Demo

Ứng dụng Demo cung cấp giao diện trực quan để kiểm thử mô hình với Webcam.

### Chạy ứng dụng

Từ thư mục gốc của dự án, chạy lệnh:

```bash
python src/demo/ui_app.py
```

### Sử dụng
-   Ứng dụng sẽ tự động mở Webcam.
-   Giao diện hiển thị 2 luồng xử lý song song:
    -   **CNN + SVM Model**
    -   **HOG + SVM Model**
-   Bảng thống kê bên phải hiển thị FPS và tỷ lệ tập trung theo thời gian thực.
-   Nhấn nút `X` trên cửa sổ để đóng ứng dụng.

---
**Nhóm thực hiện**: Nhóm 16
