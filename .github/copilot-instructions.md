# Student Engagement System - AI Coding Instructions

## 🧠 Project Overview
This project is a real-time student engagement classification system using Computer Vision and Machine Learning. It analyzes facial expressions from a webcam feed to determine if a student is "Engaged" or "Not Engaged".

**Core Architecture:**
- **Input:** Webcam video stream (OpenCV).
- **Face Detection:** Haar Cascade Classifiers (`src/face_detection`).
- **Preprocessing:** Grayscale -> Gaussian Blur -> Histogram Equalization -> Resize (128x128).
- **Feature Extraction:** Custom CNN model (outputting feature vectors).
- **Classification:** SVM (Support Vector Machine) classifier.
- **UI:** Python Tkinter application (`src/demo/ui_app.py`).

## 🏗️ Architectural Patterns

### 1. Hybrid Model Pipeline
The system uses a two-stage inference process. **Do not merge these into a single end-to-end deep learning call** unless refactoring the entire training pipeline.
- **Stage 1 (CNN):** Loads a `.h5` Keras model to extract features from preprocessed face images.
- **Stage 2 (SVM):** Loads a `.pkl` Joblib model to classify the features into one of 6 states.
- **Mapping:** The 6 states are mapped to binary "Engaged" (1) or "Not Engaged" (0) in `src/demo/utils.py`.

### 2. Preprocessing Consistency
**CRITICAL:** The preprocessing logic in `src/demo/utils.py` (inference) MUST match `src/data_processing/dataset_cleaner.py` (training).
- Always use `preprocess_roi` or equivalent logic: `Gray -> Gaussian -> HistEq -> Resize`.
- Normalization: Pixel values must be scaled to `0-1` (float32) before feeding into the CNN.

### 3. UI/Logic Separation
- `src/demo/ui_app.py`: Handles Tkinter GUI, threading, and video loop.
- `src/demo/utils.py`: Handles model loading, prediction logic, and drawing results.
- `src/face_detection/`: Isolated face detection logic.

## 🛠️ Developer Workflows

### Running the Demo
The application is designed to be run from the project root to ensure imports work correctly.
```bash
# Activate virtual environment first
python src/demo/ui_app.py
```

### Model Management
- **CNN Models:** Stored as `.h5` files in `src/demo/models/`. Load with `tf.keras.models.load_model`.
- **SVM Models:** Stored as `.pkl` files in `src/demo/models/`. Load with `joblib.load`.
- **HOG Models:** Alternative path using HOG features + SVM is supported but secondary.

### Dependency Management
- Standard `requirements.txt`.
- Key libs: `tensorflow`, `opencv-python`, `scikit-learn`, `joblib`, `tkinter` (usually built-in).

## 🚨 Conventions & Best Practices

### Path Handling
- Always use `os.path.join` and relative paths based on `__file__` to locate resources (models, cascades).
- **Do not hardcode absolute paths** (e.g., `D:\...`).

### Threading
- The UI runs on the main thread.
- Video processing and model inference should ideally not block the UI, though the current implementation might run synchronously in the update loop. Be mindful of performance.

### Label Mapping
- **0:** Bored (Not Engaged)
- **1:** Confused (Engaged)
- **2:** Drowsy (Not Engaged)
- **3:** Engaged (Engaged)
- **4:** Frustrated (Engaged)
- **5:** Looking Away (Not Engaged)

## 📂 Key Files
- `src/demo/ui_app.py`: Main entry point.
- `src/demo/utils.py`: Inference pipeline and helpers.
- `src/data_processing/dataset_cleaner.py`: Canonical preprocessing logic.
- `notebooks/student_engagement_CNNmodel.ipynb`: Reference for model training architecture.
