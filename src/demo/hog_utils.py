import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(roi):
    """
    Extracts HOG features from a Region of Interest (ROI).
    Configuration matches the training parameters in src/app/models.py:
    - Resize to 128x128
    - Grayscale
    - Orientations: 9
    - Pixels per cell: 8x8
    - Cells per block: 2x2
    - Block norm: L2-Hys
    """
    # 1. Convert to Grayscale if needed
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize to 128x128
    roi = cv2.resize(roi, (128, 128))
    
    # 3. Extract Features
    features = hog(
        roi,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    
    return features
