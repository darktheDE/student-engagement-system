"""
Utils module for Student Engagement Demo
Contains preprocessing, visualization, metrics, and helper functions
"""

from src.demo.utils_new.preprocessing import preprocess_for_cnn
from src.demo.utils_new.visualization import draw_prediction_on_frame
from src.demo.utils_new.metrics import calculate_engagement_rate, calculate_state_breakdown, calculate_agreement_rate, calculate_confidence
from src.demo.utils_new.mapping import map_prediction_to_binary, get_label_name

__all__ = [
    'preprocess_for_cnn',
    'draw_prediction_on_frame',
    'calculate_engagement_rate',
    'calculate_state_breakdown',
    'calculate_agreement_rate',
    'calculate_confidence',
    'map_prediction_to_binary',
    'get_label_name'
]
