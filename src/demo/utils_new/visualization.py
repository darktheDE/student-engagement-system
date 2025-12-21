"""
Visualization utilities for drawing predictions on frames
"""
import cv2
from src.demo.config import LABEL_MAP, BINARY_MAP


def draw_prediction_on_frame(frame, face_coords, prediction, label_map=None):
    """
    Draws bounding boxes and classification results on the frame.
    
    Args:
        frame: Input frame (numpy array)
        face_coords: Face coordinates (x, y, w, h)
        prediction: Predicted class (0-5)
        label_map: Optional custom label map (default: LABEL_MAP from config)
        
    Returns:
        numpy.ndarray: Frame with drawn results
    """
    x, y, w, h = face_coords
    
    # Use standard map if none provided
    if label_map is None:
        label_map = LABEL_MAP
    
    label = label_map.get(prediction, f"Unknown ({prediction})")
    binary_val = BINARY_MAP.get(prediction, 0)
    
    # Color: Green for Engaged (1), Red for Not Engaged (0)
    color = (0, 255, 0) if binary_val == 1 else (0, 0, 255)
    
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Draw background for text
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - 25), (x + text_w, y), color, -1)
    
    # Draw text label
    cv2.putText(frame, label, (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def draw_clean_rectangle(frame, face_coords):
    """
    Draw only a rectangle without labels
    
    Args:
        frame: Input frame
        face_coords: Face coordinates (x, y, w, h)
        
    Returns:
        numpy.ndarray: Frame with drawn rectangle
    """
    x, y, w, h = face_coords
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame
