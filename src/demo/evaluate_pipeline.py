
import os
import cv2
import sys
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.demo.core.model_manager import ModelManager
from src.demo.core.predictor import Predictor
from src.face_detection.face_detector import FaceDetector
from src.demo.config import LABEL_MAP

# Map lower case folder names to ID
# Updated to match config.py (Alphabetical Order)
FOLDER_TO_ID = {
    "looking away": 0,
    "bored": 1,
    "confused": 2,
    "drowsy": 3,
    "engaged": 4,
    "frustrated": 5
}

DATA_DIR = r"d:\HCMUTE\HCMUTE_HK5\DIPR\final\student-engagement-system\data\raw\Student-engagement-dataset"

def evaluate():
    print("="*80)
    print("             DEMO PIPELINE EVALUATION ON RAW DATASET (ALL MODELS)             ")
    print("="*80)
    
    # 1. Initialize Components
    print("\n[1] Initializing Components...")
    
    print("Loading Models...")
    try:
        model_manager = ModelManager()
        model_manager.load_all_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    predictor = Predictor(model_manager)
    
    print("Initializing Face Detector (Haar Cascade)...")
    face_detector = FaceDetector(use_dnn=False)
    
    # Storage for results
    y_true = []
    y_pred_hog = []
    y_pred_cnn_svm = []
    y_pred_cnn_pure = []
    
    # Counters
    total_images = 0
    face_detected_count = 0
    processed_count = 0
    
    start_time = time.time()
    
    print(f"\n[2] Scanning dataset at: {DATA_DIR}")
    
    # Iterate through categories
    for category in ["Engaged", "Not engaged"]:
        cat_path = os.path.join(DATA_DIR, category)
        if not os.path.exists(cat_path):
            print(f"Warning: Category folder not found: {cat_path}")
            continue
            
        print(f"\nScanning Category: {category}...")
        
        # Iterate through class folders
        for class_name in os.listdir(cat_path):
            class_path = os.path.join(cat_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            clean_name = class_name.lower().strip()
            
            if clean_name not in FOLDER_TO_ID:
                print(f"  [WARN] Skipping unknown class folder: {class_name}")
                continue
            
            label_id = FOLDER_TO_ID[clean_name]
            label_str = LABEL_MAP[label_id]
            
            # Get images
            image_files = [f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"  -> Class: '{class_name}' (ID: {label_id} - {label_str}) | Images: {len(image_files)}")
            
            # Process each image
            for img_name in image_files:
                img_path = os.path.join(class_path, img_name)
                total_images += 1
                
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                # PIPELINE STEP 1: Detect Faces
                faces = face_detector.detect_faces(frame)
                
                if len(faces) > 0:
                    # Select largest face
                    face_rect = max(faces, key=lambda f: f[2] * f[3])
                    
                    # PIPELINE STEP 2: Extract ROI (Adaptive Padding for Robustness)
                    roi = face_detector.extract_roi(frame, face_rect, padding=0, adaptive_padding=True)
                    
                    if roi is not None and roi.size > 0:
                        # PIPELINE STEP 3: Predict with ALL models
                        
                        # HOG + SVM
                        pred_hog = predictor.predict_hog_svm(roi)
                        
                        # CNN + SVM
                        pred_cnn_svm = predictor.predict_cnn_feature_svm(roi)
                        
                        # CNN Pure (Softmax)
                        pred_cnn_pure = predictor.predict_cnn_softmax(roi)
                        
                        # Only record if at least one prediction succeeded (usually all or none)
                        if pred_hog is not None: 
                            y_true.append(label_id)
                            y_pred_hog.append(pred_hog)
                            # Handle cases where CNN might fail but HOG passed, append -1 or similar if needed, 
                            # but here we assume if ROI is valid, all models run.
                            y_pred_cnn_svm.append(pred_cnn_svm if pred_cnn_svm is not None else -1)
                            y_pred_cnn_pure.append(pred_cnn_pure if pred_cnn_pure is not None else -1)
                            
                            face_detected_count += 1
                            processed_count += 1
                            
                    else:
                        pass # Face found but ROI extraction failed
                else:
                    pass # No face detected
    
    # 3. Report Results
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("                             EVALUATION RESULTS")
    print("="*80)
    print(f"Total Images: {total_images}")
    print(f"Processed: {processed_count} ({processed_count/total_images*100:.1f}%)")
    print(f"Time: {elapsed:.2f}s")
    print("-" * 80)
    
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    
    def print_report(name, y_pred, y_true):
        print(f"\n>> MODEL: {name}")
        print("-" * 40)
        
        # Filter out invalid predictions (-1)
        valid_indices = [i for i, x in enumerate(y_pred) if x != -1]
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
        
        if len(y_pred_valid) == 0:
            print("No valid predictions.")
            return

        acc = accuracy_score(y_true_valid, y_pred_valid)
        print(f"Overall Accuracy: {acc*100:.2f}%")
        print(classification_report(y_true_valid, y_pred_valid, target_names=target_names, digits=4))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_true_valid, y_pred_valid)
        print(f"{'T\\P':<4} |", end="")
        for n in target_names: print(f" {n[:4]:<5}", end="")
        print("\n" + "-----|" + "-"*6*len(target_names))
        for i, row in enumerate(cm):
            print(f"{target_names[i][:4]:<4} |", end="")
            for val in row: print(f" {val:<5}", end="")
            print()

    if len(y_true) > 0:
        print_report("HOG + SVM", y_pred_hog, y_true)
        print_report("CNN + SVM", y_pred_cnn_svm, y_true)
        print_report("Pure CNN", y_pred_cnn_pure, y_true)
    else:
        print("No results generated.")

if __name__ == "__main__":
    evaluate()
