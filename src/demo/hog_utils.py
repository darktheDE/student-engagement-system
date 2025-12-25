import cv2
import numpy as np
from skimage.feature import hog
from .utils.preprocessing import preprocess_image

def extract_hog_features(roi, visualize=False):
    """
    Extracts HOG features from a Region of Interest (ROI).
    Configuration matches the training parameters in src/app/models.py
    
    UPDATED: Now applies the same preprocessing (Blur + EQ) as training data.
    """
    # 1. Apply standardized preprocessing (Gray -> Blur -> EQ -> Resize)
    # This returns a 128x128 uint8 image
    processed_roi = preprocess_image(roi, target_size=128)
    
    # 2. Extract Features
    if visualize:
        features, hog_image = hog(
            processed_roi,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            visualize=True,
            feature_vector=True
        )
        return features, hog_image
    else:
        features = hog(
            processed_roi,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            visualize=False,
            feature_vector=True
        )
        return features
