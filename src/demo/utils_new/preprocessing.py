"""
Preprocessing utilities for face images
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.data_processing.dataset_cleaner import preprocess_roi


def preprocess_for_cnn(face_img, target_size=128):
    """
    Preprocesses the face image for CNN models.
    Wraps the centralized logic from src.data_processing.dataset_cleaner.
    
    Args:
        face_img: Face image (numpy array)
        target_size: Target size for resizing (default: 128)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for CNN input (1, 128, 128, 1)
    """
    # Use the exact same preprocessing as training
    # preprocess_roi handles: Gray -> Gaussian -> HistEq -> Resize
    processed = preprocess_roi(face_img, target_size=target_size)
    
    # Normalize to 0-1
    processed = processed.astype('float32') / 255.0
    
    # Expand dims for CNN input (1, 128, 128, 1)
    processed = np.expand_dims(processed, axis=-1)  # Add channel dim (128, 128, 1)
    processed = np.expand_dims(processed, axis=0)   # Add batch dim (1, 128, 128, 1)
    
    return processed
