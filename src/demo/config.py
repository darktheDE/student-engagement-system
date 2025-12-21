"""
Configuration file for Student Engagement Demo Application
Contains all constants, paths, and mapping configurations
"""
import os

# Get current directory for relative paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths
CNN_FEATURE_PATH = os.path.join(CURRENT_DIR, 'models', 'CNN_feature.h5')
CNN_SOFTMAX_PATH = os.path.join(CURRENT_DIR, 'models', 'CNN.h5')
SVM_PATH = os.path.join(CURRENT_DIR, 'models', 'svm_final_model.pkl')
HOG_SVM_PATH = os.path.join(CURRENT_DIR, 'models', 'hog_svm_model.pkl')

# Video Settings
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
DISPLAY_WIDTH_SPLIT = 480  # For comparison mode - 1:1 ratio
DISPLAY_HEIGHT_SPLIT = 480  # For comparison mode - 1:1 ratio
FRAME_SKIP = 2  # Process every Nth frame for performance
HISTORY_LEN = 100  # Number of predictions to keep in history

# Image Processing
TARGET_SIZE = 128  # Size for preprocessing face ROIs

# Light Detection Thresholds
LIGHT_TOO_DARK = 70   # Below this = too dark
LIGHT_TOO_BRIGHT = 200  # Above this = too bright

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
    0: 0,  # Bored -> Not Engaged
    1: 1,  # Confused -> Engaged (Active thinking)
    2: 0,  # Drowsy -> Not Engaged
    3: 1,  # Engaged -> Engaged
    4: 1,  # Frustrated -> Engaged (High arousal)
    5: 0   # Looking Away -> Not Engaged
}

# Model Configurations
MODEL_INFO = {
    'cnn_feature': {
        'name': 'CNN+SVM',
        'display_name': 'CNN+SVM',
        'short_name': 'CNN'
    },
    'cnn_softmax': {
        'name': 'CNN Thuần',
        'display_name': 'CNN Thuần',
        'short_name': 'CNN'
    },
    'hog': {
        'name': 'HOG+SVM',
        'display_name': 'HOG+SVM',
        'short_name': 'HOG'
    }
}

# UI Colors
COLOR_PRIMARY = "#2c3e50"
COLOR_SECONDARY = "#34495e"
COLOR_SUCCESS = "#2ecc71"
COLOR_DANGER = "#e74c3c"
COLOR_WARNING = "#f1c40f"
COLOR_INFO = "#3498db"
COLOR_LIGHT = "#ecf0f1"
COLOR_DARK = "#7f8c8d"
COLOR_DIVIDER = "#bdc3c7"
COLOR_BACKGROUND = "#f0f0f0"
COLOR_WHITE = "white"
COLOR_BLACK = "black"

# Face Detection Settings
FACE_DETECTION_SCALE_FACTOR = 1.3
FACE_DETECTION_MIN_NEIGHBORS = 4
