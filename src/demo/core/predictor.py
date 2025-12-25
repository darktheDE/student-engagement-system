"""
Predictor - Handles prediction logic for all models
"""
import numpy as np
from src.demo.utils.preprocessing import preprocess_for_cnn
from src.demo.hog_utils import extract_hog_features


class Predictor:
    """Handles predictions using loaded models"""
    
    def __init__(self, model_manager):
        """
        Initialize predictor with a model manager
        
        Args:
            model_manager: ModelManager instance with loaded models
        """
        self.model_manager = model_manager
    
    def predict_cnn_feature_svm(self, roi):
        """
        Predict using CNN feature extractor + SVM
        
        Args:
            roi: Face region of interest (numpy array)
            
        Returns:
            int: Predicted class (0-5) or None if error
        """
        try:
            if self.model_manager.cnn_feature_model is None or self.model_manager.svm_model is None:
                return None
            
            # Preprocess
            input_blob = preprocess_for_cnn(roi)
            
            # Extract features
            features = self.model_manager.cnn_feature_model.predict(input_blob, verbose=0)
            if len(features.shape) > 2:
                features = features.reshape(1, -1)
            
            # Classify with SVM
            pred = self.model_manager.svm_model.predict(features)[0]
            return int(pred)
            
        except Exception as e:
            print(f"Error in CNN+SVM prediction: {e}")
            return None
    
    def predict_cnn_softmax(self, roi):
        """
        Predict using pure CNN with softmax
        
        Args:
            roi: Face region of interest (numpy array)
            
        Returns:
            int: Predicted class (0-5) or None if error
        """
        try:
            if self.model_manager.cnn_softmax_model is None:
                return None
            
            # Preprocess
            input_blob = preprocess_for_cnn(roi)
            
            # Predict
            probs = self.model_manager.cnn_softmax_model.predict(input_blob, verbose=0)
            pred = np.argmax(probs, axis=1)[0]
            return int(pred)
            
        except Exception as e:
            print(f"Error in CNN softmax prediction: {e}")
            return None
    
    def predict_hog_svm(self, roi):
        """
        Predict using HOG features + SVM
        
        Args:
            roi: Face region of interest (numpy array)
            
        Returns:
            int: Predicted class (0-5) or None if error
        """
        try:
            if self.model_manager.hog_model is None:
                return None
            
            # Extract HOG features
            features = extract_hog_features(roi)
            features = features.reshape(1, -1)
            
            # Predict
            pred = self.model_manager.hog_model.predict(features)[0]
            return int(pred)
            
        except Exception as e:
            print(f"Error in HOG+SVM prediction: {e}")
            return None
    
    def predict_with_model(self, roi, model_id):
        """
        Predict using specified model
        
        Args:
            roi: Face region of interest (numpy array)
            model_id: Model identifier ('cnn_feature', 'cnn_softmax', 'hog')
            
        Returns:
            int: Predicted class (0-5) or None if error
        """
        if model_id == "cnn_feature":
            return self.predict_cnn_feature_svm(roi)
        elif model_id == "cnn_softmax":
            return self.predict_cnn_softmax(roi)
        elif model_id == "hog":
            return self.predict_hog_svm(roi)
        else:
            print(f"Unknown model_id: {model_id}")
            return None
