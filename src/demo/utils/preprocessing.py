"""
Preprocessing utilities for face images
Optimized using OpenCV
"""
import numpy as np
import sys
import os
import cv2

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def preprocess_image(face_img, target_size=128):
    """
    Base preprocessing pipeline optimized with OpenCV.
    Matches training data cleaning steps: 
    Gray -> Gaussian Blur -> Hist EQ -> Resize (NN).
    
    Returns:
        numpy.ndarray: Processed image (uint8, 2D)
    """
    # 1. Convert to Gray
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    # 2. Gaussian Blur (match training: size=5, sigma=1.0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # 3. Histogram Equalization
    equalized = cv2.equalizeHist(blurred)
    
    # 4. Resize (match training: manual resizing implied Nearest Neighbor)
    resized = cv2.resize(equalized, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    return resized

def preprocess_for_cnn(face_img, target_size=128):
    """
    Preprocesses the face image for CNN models (Normalize + Reshape).
    
    Args:
        face_img: Face image (numpy array, BGR)
        target_size: Target size for resizing (default: 128)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for CNN input (1, 128, 128, 1)
    """
    # Get base processed image
    processed_img = preprocess_image(face_img, target_size)
    
    # 5. Normalize to 0-1
    processed = processed_img.astype('float32') / 255.0
    
    # Expand dims for CNN input (1, 128, 128, 1)
    processed = np.expand_dims(processed, axis=-1)  # Add channel dim
    processed = np.expand_dims(processed, axis=0)   # Add batch dim
    
    return processed
