# Hướng Dẫn Sửa Đổi Notebook CNN cho Colab
## Student Engagement System - CNN Feature Extractor + SVM

**Ngày**: 15 Tháng 12, 2025  
**Dành cho**: Thạch (Training trên Colab)  
**Mục tiêu**: Chuyển từ CNN end-to-end → CNN Feature Extractor + SVM Classifier

---

## 📋 TÓM TẮT LUỒNG HIỆN TẠI

Theo file `flow.md`, luồng xử lý của dự án:

```
1. Dataset gốc (ảnh webcam đã label) → data/Student-engagement-dataset/
2. Face Detection (Haar Cascades) → Crop ROI (Thành/face_detection)
3. Preprocessing Pipeline (Huy/data_processing):
   - Chuyển ảnh xám
   - Gaussian Filter
   - Histogram Equalization
   - Resize 128×128
   → Output: processed/Student-engagement-dataset-clean/
   
4. CNN Feature Extraction (Thạch - notebook này):
   - Input: Ảnh đã xử lý (128×128 grayscale)
   - Bỏ bước cuối (softmax)
   - Export: feature_extractor.h5 + features.npy
   
5. SVM Training (Thạch):
   - Load features từ CNN
   - Train SVM classifier
   - Export: engagement_svm_model.pkl
```

**⚠️ Vấn đề hiện tại**: Notebook đang dùng ảnh gốc RGB 256×256, cần chuyển sang ảnh đã xử lý 128×128 grayscale.

---

## 🔧 CÁC THAY ĐỔI CẦN THỰC HIỆN

### **THAY ĐỔI 1: Cập nhật đường dẫn dataset** 
**📍 Vị trí**: Cell thứ 6 - "Define Paths"

**❌ Code hiện tại**:
```python
data_path = "/content/drive/MyDrive/Đồ Án DIP/archive/Student-engagement-processed"
```

**✅ Code mới**:
```python
# Đường dẫn đến dataset ĐÃ XỬ LÝ (grayscale, Gaussian, Histogram EQ, 128x128)
data_path = "/content/drive/MyDrive/Đồ Án DIP/Student-engagement-dataset-clean"
```

**💡 Lý do**: 
- Dataset gốc chưa qua preprocessing
- Cần dùng dataset đã được Huy xử lý (grayscale + Gaussian + Histogram + Resize 128×128)
- Tên folder theo convention: `Student-engagement-dataset-clean`

---

### **THAY ĐỔI 2: Loại bỏ xử lý ảnh trùng lặp**
**📍 Vị trí**: Cell thứ 9 - "Data Preprocessing and Split data"

**❌ Code hiện tại** (dòng 69-72):
```python
for file_name in os.listdir(subclass_dir):
    img = cv2.imread(os.path.join(subclass_dir, file_name))
    img = cv2.resize(img, img_size)                    # ← XÓA: Đã resize rồi
    img = img / 255.0                                  # ← XÓA: Normalize sau
    
    if len(os.listdir(target_test_class_dir)) != test_img_count:
        cv2.imwrite(os.path.join(target_test_class_dir, file_name), img)
    else:
        cv2.imwrite(os.path.join(target_train_class_dir, file_name), img)
```

**✅ Code mới**:
```python
for file_name in os.listdir(subclass_dir):
    img_path = os.path.join(subclass_dir, file_name)
    target_file = os.path.join(target_test_class_dir, file_name) \
                  if len(os.listdir(target_test_class_dir)) != test_img_count \
                  else os.path.join(target_train_class_dir, file_name)
    
    # CHỈ COPY FILE - Không resize/normalize vì ảnh đã xử lý sẵn
    shutil.copy(img_path, target_file)
```

**💡 Lý do**:
- Ảnh input ĐÃ qua Gaussian + Histogram EQ + Resize 128×128
- Resize lại → mất chất lượng
- Normalize 2 lần (ở đây + trong generator) → SAI!
- Chỉ cần copy file, không xử lý gì thêm

---

### **THAY ĐỔI 3: Loại bỏ normalize trong split validation**
**📍 Vị trí**: Cell thứ 12 - "Split Train into train and validation"

**❌ Code hiện tại** (dòng 116-121):
```python
for file_name in os.listdir(train_class_dir):
    img = cv2.imread(os.path.join(train_class_dir, file_name))
    if len(os.listdir(valid_class_dir)) != valid_img_count:
        cv2.imwrite(os.path.join(valid_class_dir, file_name), img)
    else:
        cv2.imwrite(os.path.join(new_class_dir, file_name), img)
```

**✅ Code mới**:
```python
for file_name in os.listdir(train_class_dir):
    img_path = os.path.join(train_class_dir, file_name)
    target_file = os.path.join(valid_class_dir, file_name) \
                  if len(os.listdir(valid_class_dir)) != valid_img_count \
                  else os.path.join(new_class_dir, file_name)
    
    # CHỈ COPY FILE - Không xử lý gì thêm
    shutil.copy(img_path, target_file)
```

**💡 Lý do**: Tương tự THAY ĐỔI 2, chỉ copy file đã xử lý sẵn.

---

### **THAY ĐỔI 4: Cập nhật ImageDataGenerator và kích thước ảnh**
**📍 Vị trí**: Cell thứ 13 - Trước "Build Resnet50 Model"

**❌ Code hiện tại**:
```python
batch_size = 32
img_size = (256, 256, 3)

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=batch_size,
        class_mode = 'categorical',
        shuffle=True)
```

**✅ Code mới**:
```python
batch_size = 32
img_size = (128, 128, 1)  # 128x128 grayscale (1 channel)

# Normalize trong generator (ảnh vẫn ở scale 0-255)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),      # ← THÊM: Đảm bảo size đúng
        color_mode='grayscale',      # ← THÊM: Đọc ảnh xám
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

val_generator = val_datagen.flow_from_directory(
        valid_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

print(train_generator.class_indices)
```

**💡 Lý do**:
- Dataset đã xử lý là grayscale → `color_mode='grayscale'`, `img_size=(128,128,1)`
- Dataset đã resize 128×128 → `target_size=(128,128)`
- Normalize bằng `rescale=1./255` trong generator (1 lần duy nhất)

---

### **THAY ĐỔI 5: Tạo 2 models - Feature Extractor + Full Model**
**📍 Vị trí**: Cell thứ 14 - "Build Resnet50 Model"

**❌ Code hiện tại**:
```python
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (1,1), activation='relu', input_shape=img_size),
    keras.layers.MaxPooling2D(3,3),
    keras.layers.Conv2D(64, (1,1), activation='relu'),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(3,3),
    keras.layers.Conv2D(128, (1,1), activation='relu'),
    keras.layers.Conv2D(128, (5,5), activation='relu'),
    keras.layers.MaxPooling2D(3,3),
    keras.layers.Conv2D(256, (1,1), activation='relu'),
    keras.layers.Conv2D(256, (5,5), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])
model.summary()
model.compile(optimizer=Adam() ,loss="categorical_crossentropy", metrics = ['accuracy'])
```

**✅ Code mới**:
```python
from tensorflow.keras import Input, Model

# ============================================================
# BƯỚC 1: Tạo Feature Extractor (không có softmax)
# ============================================================
input_layer = Input(shape=img_size)  # (128, 128, 1)

# Conv Block 1
x = keras.layers.Conv2D(32, (1,1), activation='relu', name='conv1_1x1')(input_layer)
x = keras.layers.MaxPooling2D(3,3, name='pool1')(x)

# Conv Block 2
x = keras.layers.Conv2D(64, (1,1), activation='relu', name='conv2_1x1')(x)
x = keras.layers.Conv2D(64, (3,3), activation='relu', name='conv2_3x3')(x)
x = keras.layers.MaxPooling2D(3,3, name='pool2')(x)

# Conv Block 3
x = keras.layers.Conv2D(128, (1,1), activation='relu', name='conv3_1x1')(x)
x = keras.layers.Conv2D(128, (5,5), activation='relu', name='conv3_5x5')(x)
x = keras.layers.MaxPooling2D(3,3, name='pool3')(x)

# Conv Block 4
x = keras.layers.Conv2D(256, (1,1), activation='relu', name='conv4_1x1')(x)
x = keras.layers.Conv2D(256, (5,5), activation='relu', name='conv4_5x5')(x)

# Feature Vector (512-dimensional)
x = keras.layers.Flatten(name='flatten')(x)
features = keras.layers.Dense(512, activation='relu', name='features')(x)

# Feature Extractor model (OUTPUT: 512-dim vector)
feature_extractor = Model(inputs=input_layer, outputs=features, name='Feature_Extractor')

# ============================================================
# BƯỚC 2: Tạo Full Model (có softmax) để train
# ============================================================
# Thêm classification layer
output = keras.layers.Dense(6, activation='softmax', name='classifier')(features)

# Full model (để train)
full_model = Model(inputs=input_layer, outputs=output, name='Full_Model')

# ============================================================
# SUMMARY
# ============================================================
print("="*60)
print("FEATURE EXTRACTOR (sẽ export cho SVM):")
print("="*60)
feature_extractor.summary()

print("\n" + "="*60)
print("FULL MODEL (dùng để train CNN):")
print("="*60)
full_model.summary()

# Compile chỉ full_model (dùng để train)
full_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**💡 Lý do**:
- Cần 2 models:
  - **`feature_extractor`**: Conv layers + Dense(512) - KHÔNG có softmax → Export .h5
  - **`full_model`**: Feature extractor + softmax → Dùng để train
- Đặt tên layer để dễ debug
- Feature vector 512-dim sẽ đưa vào SVM

---

### **THAY ĐỔI 6: Train full_model thay vì model**
**📍 Vị trí**: Cell thứ 15 - Training cell

**❌ Code hiện tại**:
```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[checkpoint, early_stopping]
)
```

**✅ Code mới**:
```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Checkpoint save full_model (có softmax)
checkpoint = ModelCheckpoint(
    "Student_Engagement_Full_Model.h5",  # ← Đổi tên rõ ràng
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch'
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=9,
    verbose=1,
    restore_best_weights=True
)

# Train full_model (có softmax)
history = full_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[checkpoint, early_stopping]
)
```

**💡 Lý do**: Train `full_model` (có classifier layer) để có gradient flow đúng.

---

### **THAY ĐỔI 7: THÊM CELL MỚI - Extract features cho SVM**
**📍 Vị trí**: Cell mới SAU cell training (sau cell 15)

**➕ Code mới (thêm cell mới)**:
```python
import numpy as np
import pickle

print("="*60)
print("EXTRACTING FEATURES FOR SVM TRAINING")
print("="*60)

# ============================================================
# BƯỚC 1: Extract features từ các tập dữ liệu
# ============================================================
print("\n[1/3] Extracting Train Features...")
train_generator.reset()  # Reset để đảm bảo đọc từ đầu
train_features = feature_extractor.predict(train_generator, verbose=1)
train_labels = train_generator.classes
print(f"   ✓ Train: {train_features.shape} features, {len(train_labels)} labels")

print("\n[2/3] Extracting Validation Features...")
val_generator.reset()
val_features = feature_extractor.predict(val_generator, verbose=1)
val_labels = val_generator.classes
print(f"   ✓ Val: {val_features.shape} features, {len(val_labels)} labels")

print("\n[3/3] Extracting Test Features...")
test_generator.reset()
test_features = feature_extractor.predict(test_generator, verbose=1)
test_labels = test_generator.classes
print(f"   ✓ Test: {test_features.shape} features, {len(test_labels)} labels")

# ============================================================
# BƯỚC 2: Save features và labels
# ============================================================
output_dir = "/content/drive/MyDrive/Đồ Án DIP/CNN_SVM_Output"
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("SAVING FILES")
print("="*60)

# Save features
np.save(os.path.join(output_dir, "train_features.npy"), train_features)
np.save(os.path.join(output_dir, "val_features.npy"), val_features)
np.save(os.path.join(output_dir, "test_features.npy"), test_features)
print("✓ Saved: train/val/test_features.npy")

# Save labels
np.save(os.path.join(output_dir, "train_labels.npy"), train_labels)
np.save(os.path.join(output_dir, "val_labels.npy"), val_labels)
np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)
print("✓ Saved: train/val/test_labels.npy")

# Save class_indices (mapping class name → số)
with open(os.path.join(output_dir, "class_indices.pkl"), 'wb') as f:
    pickle.dump(train_generator.class_indices, f)
print("✓ Saved: class_indices.pkl")
print(f"   Class mapping: {train_generator.class_indices}")

# ============================================================
# BƯỚC 3: Save feature extractor model
# ============================================================
feature_extractor_path = os.path.join(output_dir, "Student_Engagement_Feature_Extractor.h5")
feature_extractor.save(feature_extractor_path)
print(f"\n✓ Saved Feature Extractor: {feature_extractor_path}")

print("\n" + "="*60)
print("✅ ALL FILES READY FOR SVM TRAINING!")
print("="*60)
print("\nFiles to download:")
print("  1. Student_Engagement_Feature_Extractor.h5")
print("  2. train_features.npy + train_labels.npy")
print("  3. val_features.npy + val_labels.npy")
print("  4. test_features.npy + test_labels.npy")
print("  5. class_indices.pkl")
```

**💡 Lý do**:
- SVM cần features (vector 512-dim) thay vì ảnh
- Extract features từ train/val/test sets
- Save thành `.npy` để dễ load trong Python
- Save `feature_extractor.h5` để dùng khi inference (webcam demo)
- Save `class_indices.pkl` để biết mapping: `{'bored': 0, 'confused': 1, ...}`

---

### **THAY ĐỔI 8: Cập nhật evaluation cell**
**📍 Vị trí**: Cell sau training (cell 16 hiện tại)

**❌ Code hiện tại**:
```python
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

**✅ Code mới**:
```python
# Evaluate full_model (CNN + softmax)
test_loss, test_accuracy = full_model.evaluate(test_generator)
print("="*60)
print("CNN + SOFTMAX EVALUATION (baseline)")
print("="*60)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\n⚠️  NOTE: SVM accuracy sẽ cao hơn! (expected +3-7%)")
```

**💡 Lý do**: 
- Evaluate `full_model` để có baseline accuracy
- SVM thường cho accuracy cao hơn 3-7% với small dataset

---

## 📦 FILES CẦN EXPORT TỪ COLAB

Sau khi chạy xong notebook, download các file sau về máy:

### **Từ folder `/content/drive/MyDrive/Đồ Án DIP/CNN_SVM_Output/`**:

1. **`Student_Engagement_Feature_Extractor.h5`** (~5-10 MB)
   - Model CNN không có softmax
   - Dùng để extract features từ ảnh mới (webcam demo)

2. **`train_features.npy`** (~10-50 MB tùy số ảnh)
   - Features từ train set (mỗi ảnh → vector 512-dim)
   - Shape: `(N_train, 512)`

3. **`val_features.npy`** 
   - Features từ validation set
   - Shape: `(N_val, 512)`

4. **`test_features.npy`** 
   - Features từ test set
   - Shape: `(N_test, 512)`

5. **`train_labels.npy`** 
   - Labels của train set (array of integers: 0-5)
   - Shape: `(N_train,)`

6. **`val_labels.npy`** 
   - Labels của validation set

7. **`test_labels.npy`** 
   - Labels của test set

8. **`class_indices.pkl`** 
   - Dictionary mapping: `{'bored': 0, 'confused': 1, ...}`
   - Dùng để convert prediction số → tên class

### **(Optional) Từ folder `/content/`**:

9. **`Student_Engagement_Full_Model.h5`** (~5-10 MB)
   - Model đầy đủ có softmax (để so sánh accuracy)

---

## 🔄 QUY TRÌNH THỰC HIỆN

### **Bước 1: Chuẩn bị dataset trên Drive**
```
1. Chờ Huy hoàn thành preprocessing
2. Upload folder `Student-engagement-dataset-clean/` lên Drive:
   /content/drive/MyDrive/Đồ Án DIP/Student-engagement-dataset-clean/
   ├── Engaged/
   │   ├── confused/
   │   ├── engaged/
   │   └── frustrated/
   └── Not engaged/
       ├── bored/
       ├── drowsy/
       └── Looking Away/
```

### **Bước 2: Sửa notebook trên Colab**
```
1. Mở notebook: student_engagement_CNNmodel.ipynb
2. Áp dụng 8 THAY ĐỔI ở trên (copy-paste code)
3. Kiểm tra lại:
   - Cell 6: đường dẫn dataset đúng chưa?
   - Cell 9, 12: đã bỏ resize/normalize chưa?
   - Cell 13: đã đổi sang 128x128 grayscale chưa?
   - Cell 14: đã tạo 2 models chưa?
   - Cell mới sau 15: đã thêm extract features chưa?
```

### **Bước 3: Chạy training**
```
1. Runtime → Run all
2. Chờ training (expected: 15-25 mins với 30 epochs)
3. Kiểm tra:
   - Training accuracy cuối > 85%?
   - Validation accuracy > 80%?
   - Files đã được save trong folder CNN_SVM_Output/?
```

### **Bước 4: Download files**
```
1. Vào folder: /content/drive/MyDrive/Đồ Án DIP/CNN_SVM_Output/
2. Download TẤT CẢ 8 files (hoặc 9 files nếu có full_model)
3. Lưu vào máy: student-engagement-system/models/
4. Commit lên Git (nếu < 100MB) hoặc upload Drive riêng
```

### **Bước 5: Train SVM (notebook mới)**
```
Tạo notebook mới: student_engagement_SVM.ipynb
(Hướng dẫn chi tiết sẽ có trong file riêng)

1. Load features + labels từ .npy
2. Normalize features (StandardScaler)
3. GridSearch tìm best hyperparameters
4. Train SVM với best params
5. Evaluate trên test set
6. Save SVM model (.pkl)
```

---

## 📊 KẾT QUẢ KỲ VỌNG

### **Training Metrics**:
| Metric | Expected Value | Notes |
|--------|----------------|-------|
| Train Accuracy | 85-92% | Grayscale + preprocessing tốt |
| Val Accuracy | 80-88% | Không quá overfit |
| Test Accuracy (CNN+Softmax) | 78-85% | Baseline |
| Test Accuracy (CNN+SVM) | **83-92%** | +3-7% so với softmax |
| Training Time | 15-25 mins | 30 epochs, early stopping |
| Feature Extraction Time | 2-5 mins | Cho cả train/val/test |

### **File Sizes**:
- `Feature_Extractor.h5`: ~8 MB
- `train_features.npy`: ~20 MB (giả sử 4000 ảnh)
- `val_features.npy`: ~5 MB (1000 ảnh)
- `test_features.npy`: ~6 MB (1250 ảnh)
- Labels files: < 1 MB mỗi file
- Total: **~40-50 MB**

---

## ⚠️ LƯU Ý QUAN TRỌNG

### **1. Về Dataset Preprocessing**:
- ❌ **KHÔNG** resize/normalize trong notebook này
- ✅ Dataset đã được xử lý sẵn bởi `src/data_processing/`
- ✅ Chỉ cần copy files và split train/val/test

### **2. Về Image Size**:
- ❌ **KHÔNG** dùng 256×256 như trước
- ✅ Dùng **128×128 grayscale** theo luồng mới
- Lý do: Balance giữa accuracy và speed (30 FPS cho demo)

### **3. Về Models**:
- ❌ **KHÔNG** chỉ train 1 model rồi bỏ softmax layer
- ✅ Tạo 2 models ngay từ đầu:
  - `feature_extractor`: Không có softmax → export .h5
  - `full_model`: Có softmax → dùng để train
- Lý do: Đảm bảo architecture đúng cho inference

### **4. Về Feature Extraction**:
- ❌ **KHÔNG** dùng `model.predict()` rồi cắt bỏ layer cuối
- ✅ Dùng `feature_extractor.predict()` (model riêng biệt)
- Lý do: Tránh lỗi shape và dễ maintain code

### **5. Về Files Export**:
- ✅ Save features dạng `.npy` (NumPy format) - nhanh và compact
- ✅ Save class_indices dạng `.pkl` (Pickle) - giữ nguyên dictionary
- ✅ Save model dạng `.h5` (Keras HDF5) - load dễ dàng
- ❌ **KHÔNG** save dạng `.csv` (quá chậm và tốn dung lượng)

---

## 🐛 TROUBLESHOOTING

### **Lỗi: "Input shape mismatch"**
```
Symptom: ValueError: Input 0 of layer "..." is incompatible with the layer
Solution: 
  - Kiểm tra img_size = (128, 128, 1)
  - Kiểm tra color_mode='grayscale' trong generator
  - Kiểm tra target_size=(128,128) trong generator
```

### **Lỗi: "No images found"**
```
Symptom: Found 0 images belonging to X classes
Solution:
  - Kiểm tra đường dẫn data_path có đúng không
  - Kiểm tra cấu trúc folder: data_path/Engaged/confused/*.jpg
  - Kiểm tra đã upload dataset lên Drive chưa
```

### **Lỗi: "cv2.imwrite failed"**
```
Symptom: Error in Cell 9 hoặc 12 khi save ảnh
Solution:
  - Thay cv2.imread + cv2.imwrite → shutil.copy
  - Không cần đọc ảnh vào memory, chỉ copy file
```

### **Lỗi: "Out of Memory"**
```
Symptom: ResourceExhaustedError during training
Solution:
  - Giảm batch_size: 32 → 16 hoặc 8
  - Restart runtime: Runtime → Restart runtime
  - Dùng GPU: Runtime → Change runtime type → GPU
```

### **Accuracy quá thấp (< 70%)**
```
Possible causes:
  1. Dataset chưa được xử lý đúng (vẫn dùng ảnh gốc?)
  2. Generator config sai (RGB thay vì grayscale?)
  3. Normalize 2 lần (trong split + generator)
  4. Augmentation bị tắt hoặc quá mạnh
  
Solution:
  - Verify dataset path: phải là "...-clean" folder
  - Check generator: color_mode='grayscale', rescale=1./255
  - Bỏ normalize trong split (Cell 9, 12)
```

---

## 📚 TÀI LIỆU THAM KHẢO

### **Papers**:
1. **CNN Feature Extraction**: Razavian et al. (2014) - "CNN Features off-the-shelf"
2. **SVM for Small Datasets**: Cortes & Vapnik (1995) - "Support-vector networks"
3. **Student Engagement**: Whitehill et al. (2014) - "Facial Expression Recognition"

### **Code Examples**:
- Keras Functional API: https://keras.io/guides/functional_api/
- Feature Extraction: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
- SVM with CNN features: sklearn + keras integration

### **Related Files**:
- `docs/flow.md` - Luồng xử lý tổng thể
- `docs/CHAT_SESSION_SUMMARY.md` - Chi tiết workflow
- `src/data_processing/dataset_cleaner.py` - Preprocessing code
- `src/face_detection/__init__.py` - Face detection code

---

## ✅ CHECKLIST TRƯỚC KHI CHẠY

- [ ] Dataset đã được Huy preprocessing xong
- [ ] Dataset đã upload lên Drive đúng đường dẫn
- [ ] Đã sửa Cell 6: đường dẫn dataset
- [ ] Đã sửa Cell 9: bỏ resize/normalize, dùng shutil.copy
- [ ] Đã sửa Cell 12: bỏ cv2.imwrite, dùng shutil.copy
- [ ] Đã sửa Cell 13: img_size=(128,128,1), color_mode='grayscale'
- [ ] Đã sửa Cell 14: tạo 2 models (feature_extractor + full_model)
- [ ] Đã sửa Cell 15: train full_model
- [ ] Đã thêm Cell mới: extract features + save files
- [ ] Đã sửa Cell evaluation: dùng full_model.evaluate()
- [ ] Runtime đã set GPU (recommended)
- [ ] Đã mount Google Drive

---

## 🎯 TỔNG KẾT

### **Các thay đổi chính**:
1. ✅ Dataset path → processed dataset (128×128 grayscale)
2. ✅ Bỏ xử lý trùng lặp (resize/normalize) → chỉ copy files
3. ✅ Generator config → grayscale, 128×128, rescale=1./255
4. ✅ Tạo 2 models → feature_extractor + full_model
5. ✅ Train full_model (có softmax)
6. ✅ Extract features → save .npy files
7. ✅ Export feature_extractor.h5 + class_indices.pkl

### **Kết quả đạt được**:
- ✅ Feature extractor .h5 để dùng cho inference (webcam demo)
- ✅ Features .npy để train SVM
- ✅ Baseline accuracy (CNN+softmax) để so sánh
- ✅ Chuẩn bị đầy đủ cho bước tiếp theo (SVM training)

### **Bước tiếp theo**:
1. Chạy notebook này trên Colab
2. Download 8-9 files về máy
3. Tạo notebook mới: `student_engagement_SVM.ipynb`
4. Train SVM với features đã extract
5. So sánh accuracy: CNN+Softmax vs CNN+SVM
6. Integrate vào demo (webcam real-time)

---

**Người soạn**: GitHub Copilot  
**Ngày**: 15/12/2025  
**Version**: 1.0  
**Status**: ✅ READY TO IMPLEMENT

---

## 📞 LIÊN HỆ & HỖ TRỢ

- **Thành** (Face Detection): Hỗ trợ xử lý ROI và face detection pipeline
- **Huy** (Preprocessing): Hỗ trợ về dataset xử lý và format ảnh
- **Thạch** (CNN + SVM): Implement notebook này và train SVM

**Lưu ý**: Nếu có vấn đề khi chạy notebook, check lại các THAY ĐỔI từ 1-8 và so sánh với code hiện tại. Mọi thay đổi đều có **💡 Lý do** chi tiết để hiểu tại sao cần sửa.
