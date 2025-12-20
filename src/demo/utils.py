import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from src.data_processing.dataset_cleaner import preprocess_roi

def load_models(cnn_path, svm_path):
    """
    Loads the CNN feature extractor and SVM classifier.
    """
    print(f"Loading CNN model from: {cnn_path}")
    if not os.path.exists(cnn_path):
        raise FileNotFoundError(f"CNN model not found at {cnn_path}")
    
    cnn_model = tf.keras.models.load_model(cnn_path)
    # Re-compile to avoid warnings if optimizer state is not needed for inference
    cnn_model.compile(jit_compile=False) 
    
    print(f"Loading SVM model from: {svm_path}")
    if not os.path.exists(svm_path):
        raise FileNotFoundError(f"SVM model not found at {svm_path}")
        
    # Use joblib for loading sklearn models (more robust than pickle)
    svm_model = joblib.load(svm_path)
        
    return cnn_model, svm_model

def load_hog_model(model_path):
    """
    Loads the HOG SVM model.
    """
    print(f"Loading HOG SVM model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"HOG model not found at {model_path}")
        
    model = joblib.load(model_path)
    return model

def custom_preprocess(face_img, target_size=128):
    """
    Preprocesses the face image for the model.
    Wraps the centralized logic from src.data_processing.dataset_cleaner.
    """
    # Use the exact same preprocessing as training
    # preprocess_roi handles: Gray -> Gaussian -> HistEq -> Resize
    processed = preprocess_roi(face_img, target_size=target_size)
    
    # Expand dims for CNN input (1, 128, 128, 1)
    # The model likely expects a batch dimension and channel dimension
    processed = processed.astype('float32') / 255.0 # Normalize 0-1
    processed = np.expand_dims(processed, axis=-1)  # Add channel dim (128, 128, 1)
    processed = np.expand_dims(processed, axis=0)   # Add batch dim (1, 128, 128, 1)
    
    return processed

# Label Mapping based on alphabetic order of keys in ENGAGEMENT_STATES
# 0: bored, 1: confused, 2: drowsy, 3: engaged, 4: frustrated, 5: looking away
LABEL_MAP = {
    0: "Bored",
    1: "Confused",
    2: "Drowsy",
    3: "Engaged",
    4: "Frustrated",
    5: "Looking Away"
}

# Binary Group Mapping (1 = Engaged/Positive, 0 = Not Engaged/Negative)
BINARY_MAP = {
    0: 0, # Bored -> Not Engaged
    1: 1, # Confused -> Engaged (Active thinking)
    2: 0, # Drowsy -> Not Engaged
    3: 1, # Engaged -> Engaged
    4: 1, # Frustrated -> Engaged (High arousal)
    5: 0  # Looking Away -> Not Engaged
}

def map_prediction_to_binary(prediction):
    """Maps the multi-class prediction integer to a binary 0/1 score."""
    return BINARY_MAP.get(prediction, 0)

def draw_results(frame, face_coords, prediction, label_map=None):
    """
    Draws bounding boxes and classification results on the frame.
    """
    x, y, w, h = face_coords
    
    # Use standard map if none provided
    if label_map is None:
        label_map = LABEL_MAP
        
    label = label_map.get(prediction, f"Unknown ({prediction})")
    binary_val = BINARY_MAP.get(prediction, 0)
    
    # Color: Green for Engaged-like (1), Red for Not Engaged-like (0)
    color = (0, 255, 0) if binary_val == 1 else (0, 0, 255)
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Draw background for text
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - 25), (x + text_w, y), color, -1)
    
    # Draw text
    cv2.putText(frame, label, (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame
