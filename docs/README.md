# Student Engagement Demo App

Đây là ứng dụng demo thời gian thực (Real-time Demo) cho hệ thống đánh giá mức độ tập trung của sinh viên.

## 🚀 Quy Trình Xử Lý (Pipeline)

Ứng dụng hoạt động theo luồng dữ liệu như sau:

1.  **Input**: Ảnh từ Webcam (được resize về `640x480`).
2.  **Face Detection**: Sử dụng `src.face_detection.FaceDetector` (Haar Cascade) để tìm khuôn mặt.
3.  **ROI Extraction**: Cắt vùng ảnh khuôn mặt từ khung hình gốc.
4.  **Preprocessing (Quan Trọng)**:
    *   Ảnh vùng mặt được đưa vào hàm `custom_preprocess` trong `src/demo/utils.py`.
    *   Hàm này **GỌI LẠI TRỰC TIẾP** hàm `preprocess_roi` từ module `src.data_processing.dataset_cleaner`.
    *   **Các bước xử lý**:
        1.  Chuyển sang ảnh xám (Grayscale).
        2.  Lọc nhiễu Gaussian (`gaussian_filter`).
        3.  Cân bằng Histogram (`histogram_equalization`).
        4.  Resize về kích thước chuẩn `128x128`.
    *   *Điều này đảm bảo dữ liệu đầu vào của Demo hoàn toàn khớp với dữ liệu đã dùng để train model.*
5.  **Feature Extraction**: Đưa ảnh đã xử lý (128x128) vào mô hình **CNN** (`CNN_feature.h5`) để trích xuất đặc trưng.
6.  **Classification**: Vector đặc trưng được đưa vào **SVM** (`svm_final_model.pkl`) để dự đoán:
    *   `1`: Engaged (Tập trung)
    *   `0`: Not Engaged (Không tập trung)
7.  **Visualization**: Vẽ khung và hiển thị kết quả lên màn hình.

## 📁 Cấu Trúc

*   `app.py`: Logic chính của ứng dụng (Vòng lặp Webcam, FPS control, giao diện).
*   `utils.py`: Chứa các hàm hỗ trợ load model và tiền xử lý.
*   `models/`: Chứa file `CNN_feature.h5` và `svm_final_model.pkl`.

## ✅ Tái Sử Dụng Code

Ứng dụng này tuân thủ nguyên tắc tái sử dụng code bằng cách import các module đã định nghĩa trước đó:
*   Sử dụng `src.face_detection` cho việc phát hiện khuôn mặt.
*   Sử dụng `src.data_processing` cho toàn bộ quy trình xử lý ảnh.
