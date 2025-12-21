"""
Core module for Student Engagement Demo
Contains model management, prediction, and video processing logic
"""

from src.demo.core.model_manager import ModelManager
from src.demo.core.predictor import Predictor
from src.demo.core.video_processor import VideoProcessor

__all__ = ['ModelManager', 'Predictor', 'VideoProcessor']
