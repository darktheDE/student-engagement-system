# Pipeline Tối Ưu cho CNN Feature Extraction + SVM Classification

## 📋 Tổng quan

**Kiến trúc:** Hybrid CNN-SVM
- **CNN**: Trích xuất đặc trưng (Feature Extractor)
- **SVM**: Phân loại (Classifier)

## 🔄 Luồng xử lý dữ liệu

### 1. Face Detection & ROI Extraction (Cắt mặt)

```
Ảnh gốc (BGR)
    ↓
Face Detection (Haar Cascade / DNN)
    ↓
Extract ROI (vùng khuôn mặt)
    ↓
ROI đã cắt
```

#### Chi tiết:

```python
from src.face_detection import FaceDetector

detector = FaceDetector(use_dnn=False)

# Detect faces
faces = detector.detect_faces(image)

# Extract ROI từng khuôn mặt
for face_coords in faces:
    x, y, w, h = face_coords
    
    # Cắt ROI với padding
    roi = detector.extract_roi(
        image, 
        face_coords,
        padding=10,           # Thêm padding 10px
        adaptive_padding=True # Tự động điều chỉnh padding
    )
```

**Quan trọng:**
- ✅ **Padding**: Thêm 5-10% viền xung quanh mặt để không cắt mất phần rìa
- ✅ **Validation**: Có thể dùng eye detection để validate ROI
- ✅ **Quality Check**: Dùng `assess_face_quality()` để loại bỏ ảnh mờ/tối

### 2. Tiền xử lý ảnh (Preprocessing)

```
ROI đã cắt (BGR) 
    ↓
Chuyển BGR → RGB
    ↓
Resize về 128x128 (hoặc 96x96, 224x224)
    ↓
Normalize về [0, 1]
    ↓
Ảnh sẵn sàng cho CNN
```

#### Chi tiết từng bước:

**Bước 1: BGR → RGB**
```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```
- ✅ Giữ nguyên thông tin màu (3 channels)
- ✅ Màu da chứa thông tin cảm xúc: đỏ mặt, xanh xao, hồng hào

**Bước 2: Resize**
```python
resized = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_AREA)
```
- ✅ **128x128** (khuyến nghị) - cân bằng giữa chi tiết và tốc độ
- Các lựa chọn khác:
  - 96x96: Nhanh hơn, ít chi tiết hơn
  - 224x224: Chi tiết hơn, chậm hơn (ResNet, VGG)
  - ❌ 48x48: QUÁ NHỎ, mất nhiều thông tin

**Bước 3: Normalize**
```python
normalized = resized.astype('float32') / 255.0
```
- ✅ Chuyển từ [0, 255] → [0, 1]
- ✅ Giúp CNN hội tụ nhanh hơn
- ✅ Chuẩn hóa dữ liệu đầu vào

**Optional: Gaussian Filter (giảm nhiễu)**
```python
# Chỉ dùng nếu ảnh có nhiễu nhiều
img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), 0)
```

### 2. Trích xuất đặc trưng (CNN)

```
Ảnh RGB 128x128x3
    ↓
[CNN Layers]
Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → ...
    ↓
Flatten / Global Average Pooling
    ↓
Feature Vector (ví dụ: 512 features)
```

#### Kiến trúc CNN đề xuất:

**Option 1: CNN đơn giản (Custom)**
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5)
    # KHÔNG thêm Dense cuối (SVM sẽ làm)
])
```

**Option 2: Transfer Learning (Khuyến nghị)**
```python
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,  # Bỏ classification layer
    weights='imagenet',
    pooling='avg'       # Global Average Pooling
)

# Freeze base model
base_model.trainable = False

# Feature extractor
features = base_model.predict(X_train)  # Shape: (n_samples, 1280)
```

**Models phù hợp:**
- ✅ **MobileNetV2**: Nhanh, nhẹ, chính xác cao
- ✅ **EfficientNetB0**: Cân bằng tốt
- ✅ **ResNet50**: Mạnh nhưng nặng hơn
- VGG16: Nặng, không khuyến nghị

### 3. Phân loại (SVM)

```
Feature Vector từ CNN
    ↓
Chuẩn hóa features (StandardScaler)
    ↓
SVM Classifier
    ↓
Nhãn dự đoán (7 classes)
```

#### Cài đặt SVM:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Trích xuất features từ CNN
X_train_features = cnn_model.predict(X_train)  # Shape: (n, 512)
X_test_features = cnn_model.predict(X_test)

# 2. Chuẩn hóa features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# 3. Train SVM
svm = SVC(
    kernel='rbf',      # RBF kernel (khuyến nghị)
    C=1.0,             # Regularization
    gamma='scale',     # Auto
    class_weight='balanced',  # Xử lý imbalanced data
    probability=True   # Cho phép predict_proba
)

svm.fit(X_train_scaled, y_train)

# 4. Dự đoán
y_pred = svm.predict(X_test_scaled)
```

**Tham số SVM quan trọng:**
- `kernel='rbf'`: Tốt cho non-linear data
- `C=1.0`: Thử grid search [0.1, 1, 10, 100]
- `gamma='scale'`: Thử grid search ['scale', 'auto', 0.01, 0.1]
- `class_weight='balanced'`: Quan trọng nếu dataset không cân bằng

## 📊 Pipeline hoàn chỉnh

### Code đầy đủ:

```python
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===== 1. FACE DETECTION & ROI EXTRACTION =====
from src.face_detection import FaceDetector

def extract_face_roi(img_path, detector):
    """Phát hiện và cắt khuôn mặt từ ảnh"""
    # Đọc ảnh
    img = cv2.imread(img_path)
    
    # Detect faces
    faces = detector.detect_faces(img)
    
    if len(faces) == 0:
        return None  # Không tìm thấy khuôn mặt
    
    # Lấy khuôn mặt lớn nhất (giả sử là chủ thể chính)
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    
    # Extract ROI với padding
    roi = detector.extract_roi(
        img, 
        largest_face,
        padding=10,
        adaptive_padding=True
    )
    
    return roi

# ===== 2. PREPROCESSING =====
def preprocess_roi(roi, target_size=128):
    """Tiền xử lý ROI cho CNN-SVM"""
    # BGR → RGB
    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Resize
    resized = cv2.resize(img_rgb, (target_size, target_size), 
                        interpolation=cv2.INTER_AREA)
    
    # Normalize
    normalized = resized.astype('float32') / 255.0
    
    return normalized

# ===== 2. CNN FEATURE EXTRACTOR =====
def build_feature_extractor(input_shape=(128, 128, 3)):
    """Xây dựng CNN để trích xuất đặc trưng"""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False
    return base_model

# ===== 3. TRAINING PIPELINE =====
# Khởi tạo face detector
detector = FaceDetector(use_dnn=False)

# Load và preprocess data
X = []  # List các ảnh đã preprocess
y = []  # List labels

for img_path, label in dataset:
    # Bước 1: Cắt ROI (khuôn mặt)
    roi = extract_face_roi(img_path, detector)
    
    if roi is None:
        print(f"Không tìm thấy khuôn mặt: {img_path}")
        continue  # Skip ảnh không có mặt
    
    # Bước 2: Preprocess ROI
    img_processed = preprocess_roi(roi)
    
    X.append(img_processed)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Trích xuất features bằng CNN
cnn_extractor = build_feature_extractor()
X_train_features = cnn_extractor.predict(X_train)
X_test_features = cnn_extractor.predict(X_test)

# Chuẩn hóa features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Train SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale', 
          class_weight='balanced', probability=True)
svm.fit(X_train_scaled, y_train)

# Đánh giá
y_pred = svm.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# ===== 4. INFERENCE =====
def predict_image(img_path, detector, cnn_model, scaler, svm_model):
    """Dự đoán một ảnh mới"""
    # Bước 1: Extract ROI
    roi = extract_face_roi(img_path, detector)
    
    if roi is None:
        return None, None  # Không tìm thấy khuôn mặt
    
    # Bước 2: Preprocess ROI
    img = preprocess_roi(roi)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Bước 3: Extract features bằng CNN
    features = cnn_model.predict(img)
    
    # Bước 4: Scale features
    features_scaled = scaler.transform(features)
    
    # Bước 5: Predict bằng SVM
    prediction = svm_model.predict(features_scaled)[0]
    proba = svm_model.predict_proba(features_scaled)[0]
    
    return prediction, proba
```

## 🎯 So sánh với pipeline cũ

| Bước | Pipeline Cũ (48x48, Grayscale) | Pipeline Mới (128x128, RGB) |
|------|--------------------------------|------------------------------|
| **Input** | BGR → Gray | BGR → RGB |
| **Kích thước** | 48x48x1 | 128x128x3 |
| **Gaussian** | ✅ (size=5) | ⚠️ Optional (size=3) |
| **Histogram EQ** | ✅ | ❌ (thay = Normalize) |
| **Normalize** | ❌ | ✅ /255.0 |
| **Thông tin màu** | ❌ Mất | ✅ Giữ nguyên |
| **Chi tiết** | ⚠️ Ít (48x48) | ✅ Nhiều (128x128) |
| **Độ chính xác** | ~70-75% | ~85-90% (ước tính) |

## 📈 Hyperparameter Tuning

### Grid Search cho SVM:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(
    SVC(class_weight='balanced', probability=True),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)
best_svm = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")
```

## 💡 Tips để cải thiện

1. **Data Augmentation** (cho CNN):
   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   
   datagen = ImageDataGenerator(
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
       horizontal_flip=True,
       zoom_range=0.2
   )
   ```

2. **Fine-tune CNN** (sau khi train SVM):
   ```python
   base_model.trainable = True
   # Train thêm vài epoch với learning rate thấp
   ```

3. **Ensemble Models**:
   - Kết hợp nhiều SVM với các kernel khác nhau
   - Voting classifier

4. **Class balancing**:
   - SMOTE cho features
   - Class weights trong SVM

## 🎯 Kết quả mong đợi

| Metric | Giá trị mục tiêu |
|--------|------------------|
| **Accuracy** | 85-90% |
| **F1-Score (macro)** | 83-88% |
| **Training time** | 10-30 phút (GPU) |
| **Inference time** | <100ms/image |

## 📚 Tài liệu tham khảo

- MobileNetV2: https://arxiv.org/abs/1801.04381
- SVM for image classification: https://scikit-learn.org/stable/modules/svm.html
- CNN-SVM hybrid: https://ieeexplore.ieee.org/document/8308186

---

**Lưu ý:** Pipeline này được tối ưu cho bài toán Student Engagement với 7 classes (confused, engaged, frustrated, bored, drowsy, looking away, + 1 class nữa?). Điều chỉnh hyperparameters dựa trên kết quả thực nghiệm.
