# Student Engagement System

Hệ thống nhận diện trạng thái tập trung của sinh viên sử dụng computer vision.

**Đồ án xử lý ảnh số - Nhóm 16**

## Mô tả

Dự án sử dụng face detection và emotion classification để phát hiện mức độ tập trung của sinh viên qua biểu cảm khuôn mặt. Dataset được phân loại thành 2 nhóm chính:
- **Engaged**: confused, engaged, frustrated
- **Not engaged**: bored, drowsy, Looking Away

## Cấu trúc thư mục

```
student-engagement-system/
├── configs/                    # File cấu hình
│   └── config.yaml
├── data/                       # Dữ liệu (ignored trong git)
│   ├── raw/                    # Dataset gốc
│   ├── processed/              # Dataset đã xử lý
│   └── models/                 # Model đã train
├── docs/                       # Tài liệu dự án
├── notebooks/                  # Jupyter notebooks
├── sample/                     # Dữ liệu mẫu (không bị ignore)
├── src/                        # Mã nguồn chính
│   ├── data_processing/        # Xử lý và lọc ảnh
│   ├── face_detection/         # Phát hiện khuôn mặt
│   ├── emotion_recognition/    # Nhận diện cảm xúc
│   ├── engagement_classifier/  # Phân loại mức độ tập trung
│   └── visualization/          # Trực quan hóa kết quả
├── tests/                      # Unit tests
└── utils/                      # Utilities
```

## Cài đặt

```bash
# Clone repository
git clone https://github.com/darktheDE/student-engagement-system
cd student-engagement-system

# Cài đặt dependencies (tạo file requirements.txt nếu chưa có)
pip install opencv-python numpy scipy matplotlib
```

## Sử dụng

### Xử lý dataset
```python
# Chạy pipeline xử lý ảnh (Gaussian filter + grayscale)
python -m src.data_processing
```

### Face detection
```python
from src.face_detection import FaceDetector

detector = FaceDetector(use_dnn=False)
faces = detector.detect_faces(image)
rois = detector.detect_and_extract(image, target_size=(256, 256))
```

## Công nghệ

- **OpenCV**: Face detection (Haar Cascades + DNN)
- **NumPy**: Xử lý ma trận ảnh
- **SciPy**: Custom image filters
- **Matplotlib**: Visualization

## Tác giả

Nhóm 16 - HCMUTE

## License

MIT License
