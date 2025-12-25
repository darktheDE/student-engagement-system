"""
Video Processor - Handles camera and video frame processing
"""
import cv2
import time
import numpy as np
from collections import deque, Counter
from src.demo.config import (WEBCAM_WIDTH, WEBCAM_HEIGHT, FRAME_SKIP, HISTORY_LEN,
                      LIGHT_TOO_DARK, LIGHT_TOO_BRIGHT, PREDICTION_SMOOTHING_WINDOW)


class VideoProcessor:
    """Manages video capture, processing, and face detection"""
    
    def __init__(self, detector):
        """
        Initialize video processor
        
        Args:
            detector: FaceDetector instance
        """
        self.detector = detector
        self.cap = None
        self.frame_count = 0
        self.current_frame = None
        self.current_faces = []
        self.brightness_adjust = 0
        self.brightness_adjust = 0
        
        # FPS tracking
        self.fps_start = time.time()
        self.fps_counter = 0
        self.fps_val = 0
        
        # History for predictions
        self.history = {}
        
        # Processing times
        self.processing_times = {}
        
        # Predictions storage
        self.predictions = {}
        
        # Smoothing Buffer (Stores last N raw predictions)
        self.smooth_buffer = {}
    
    def initialize_camera(self):
        """
        Initialize webcam with optimized settings
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            if not self.cap.isOpened():
                print("❌ Webcam not found")
                return False
            
            print("🎥 Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return False
    
    def initialize_history(self, model_ids):
        """
        Initialize prediction history for specified models
        
        Args:
            model_ids: List of model IDs to track
        """
        for model_id in model_ids:
            self.history[model_id] = {
                'binary': deque(maxlen=HISTORY_LEN),
                'raw': deque(maxlen=HISTORY_LEN)
            }
            # Initialize smoothing buffer for this model
            self.smooth_buffer[model_id] = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)
    
    def read_frame(self):
        """
        Read and preprocess a frame from camera
        
        Returns:
            numpy.ndarray: Processed frame or None if error
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Apply manual brightness adjustment
        if self.brightness_adjust != 0:
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=self.brightness_adjust)
        
        self.current_frame = frame.copy()
        self.frame_count += 1
        
        return frame
    
    def should_process_frame(self):
        """
        Determine if current frame should be processed (frame skipping)
        
        Returns:
            bool: True if frame should be processed
        """
        return self.frame_count % FRAME_SKIP == 0
    
    def detect_faces(self, frame):
        """
        Detect faces in frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            list: List of face rectangles [(x, y, w, h), ...]
        """
        self.current_faces = self.detector.detect_faces(frame)
        return self.current_faces
    
    def get_largest_face(self):
        """
        Get the largest detected face
        
        Returns:
            tuple: (x, y, w, h) or None if no faces
        """
        if len(self.current_faces) == 0:
            return None
        
        return max(self.current_faces, key=lambda f: f[2] * f[3])
    
    def extract_face_roi(self, frame, face_rect):
        """
        Extract face ROI from frame
        
        Args:
            frame: Input frame
            face_rect: Face rectangle (x, y, w, h)
            
        Returns:
            numpy.ndarray: Face ROI or None if error
        """
        try:
            # Use adaptive padding (8%) to be robust to webcam resolution differences
            roi = self.detector.extract_roi(frame, face_rect, padding=0, adaptive_padding=True)
            if roi.size == 0:
                return None
            return roi
        except Exception as e:
            print(f"Error extracting ROI: {e}")
            return None
    
    def calculate_light_quality(self):
        """
        Calculate lighting quality of current frame
        
        Returns:
            str: Light quality status
        """
        if self.current_frame is None or len(self.current_faces) == 0:
            return "N/A"
        
        try:
            face_rect = self.get_largest_face()
            if face_rect is None:
                return "N/A"
            
            x, y, w, h = face_rect
            face_roi = self.current_frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return "N/A"
            
            # Convert to grayscale and calculate mean brightness
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray)
            
            if mean_val < LIGHT_TOO_DARK:
                return "⚠️ Quá tối"
            elif mean_val > LIGHT_TOO_BRIGHT:
                return "⚠️ Quá sáng"
            else:
                return "✓ Tốt"
        except Exception as e:
            print(f"Error calculating light quality: {e}")
            return "N/A"
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_start >= 1.0:
            self.fps_val = self.fps_counter
            self.fps_counter = 0
            self.fps_start = time.time()
    
    def get_fps(self):
        """Get current FPS value"""
        return self.fps_val
    
    def update_with_smoothing(self, model_id, raw_prediction, map_binary_func):
        """
        Update prediction, smoothing buffer, and history
        
        Args:
            model_id: Model identifier
            raw_prediction: Raw instantaneous prediction
            map_binary_func: Function to map prediction to binary value
        """
        # 1. Add raw prediction to smoothing buffer
        if model_id in self.smooth_buffer:
            self.smooth_buffer[model_id].append(raw_prediction)
            
        # 2. Get smoothed result (Voting)
        smoothed_pred = self.get_smoothed_prediction(model_id)
        if smoothed_pred is None:
            smoothed_pred = raw_prediction
            
        # 3. Calculate binary value based on SMOOTHED result
        binary_value = map_binary_func(smoothed_pred)
        
        # 4. Update current prediction (for UI)
        self.predictions[model_id] = smoothed_pred
        
        # 5. Update History (for Metrics) with SMOOTHED data
        if model_id in self.history:
            self.history[model_id]['raw'].append(smoothed_pred)
            self.history[model_id]['binary'].append(binary_value)
    
    def get_smoothed_prediction(self, model_id):
        """
        Get smoothed prediction using majority voting from buffer
        
        Args:
            model_id: Model identifier
            
        Returns:
            int: Smoothed prediction class or current prediction if buffer empty
        """
        if model_id not in self.smooth_buffer or len(self.smooth_buffer[model_id]) == 0:
            return self.predictions.get(model_id, None)
            
        # Standard Majority voting
        votes = Counter(self.smooth_buffer[model_id])
        smoothed_pred = votes.most_common(1)[0][0]
        return smoothed_pred
    
    def set_brightness(self, value):
        """
        Set brightness adjustment value
        
        Args:
            value: Brightness adjustment (-50 to +50)
        """
        self.brightness_adjust = value
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            print("Camera released")
