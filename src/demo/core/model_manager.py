"""
Model Manager - Handles loading and management of ML models
"""
import os
import joblib
import tensorflow as tf
from src.demo.config import CNN_FEATURE_PATH, CNN_SOFTMAX_PATH, SVM_PATH, HOG_SVM_PATH


class ModelManager:
    """Manages loading and storage of all ML models"""
    
    def __init__(self):
        self.cnn_feature_model = None
        self.svm_model = None
        self.cnn_softmax_model = None
        self.hog_model = None
        self.available_models = {}
    
    def load_cnn_feature_svm(self):
        """
        Loads the CNN feature extractor and SVM classifier.
        Returns: (cnn_model, svm_model) tuple or raises exception
        """
        print(f"Loading CNN model from: {CNN_FEATURE_PATH}")
        if not os.path.exists(CNN_FEATURE_PATH):
            raise FileNotFoundError(f"CNN model not found at {CNN_FEATURE_PATH}")
        
        cnn_model = tf.keras.models.load_model(CNN_FEATURE_PATH)
        cnn_model.compile(jit_compile=False)
        
        print(f"Loading SVM model from: {SVM_PATH}")
        if not os.path.exists(SVM_PATH):
            raise FileNotFoundError(f"SVM model not found at {SVM_PATH}")
        
        svm_model = joblib.load(SVM_PATH)
        
        self.cnn_feature_model = cnn_model
        self.svm_model = svm_model
        self.available_models['cnn_feature'] = {'name': 'CNN+SVM', 'loaded': True}
        
        return cnn_model, svm_model
    
    def load_cnn_softmax(self):
        """
        Loads the pure CNN model with softmax output.
        Returns: cnn_model or None if not found
        """
        if not os.path.exists(CNN_SOFTMAX_PATH):
            print(f"⚠ Pure CNN not found at {CNN_SOFTMAX_PATH}")
            self.available_models['cnn_softmax'] = {'name': 'CNN Thuần', 'loaded': False}
            return None
        
        print(f"Loading Pure CNN from: {CNN_SOFTMAX_PATH}")
        cnn_model = tf.keras.models.load_model(CNN_SOFTMAX_PATH)
        cnn_model.compile(jit_compile=False)
        
        self.cnn_softmax_model = cnn_model
        self.available_models['cnn_softmax'] = {'name': 'CNN Thuần', 'loaded': True}
        
        return cnn_model
    
    def load_hog_svm(self):
        """
        Loads the HOG + SVM model.
        Returns: hog_model or raises exception
        """
        print(f"Loading HOG SVM model from: {HOG_SVM_PATH}")
        if not os.path.exists(HOG_SVM_PATH):
            raise FileNotFoundError(f"HOG model not found at {HOG_SVM_PATH}")
        
        hog_model = joblib.load(HOG_SVM_PATH)
        
        self.hog_model = hog_model
        self.available_models['hog'] = {'name': 'HOG+SVM', 'loaded': True}
        
        return hog_model
    
    def load_all_models(self):
        """
        Attempts to load all available models.
        Returns: dict of loaded models status
        """
        results = {}
        
        # Load CNN + SVM (required)
        try:
            self.load_cnn_feature_svm()
            results['cnn_feature'] = 'success'
        except Exception as e:
            results['cnn_feature'] = f'error: {e}'
            raise
        
        # Load Pure CNN (optional)
        try:
            model = self.load_cnn_softmax()
            results['cnn_softmax'] = 'success' if model else 'not_found'
        except Exception as e:
            results['cnn_softmax'] = f'error: {e}'
        
        # Load HOG + SVM (required)
        try:
            self.load_hog_svm()
            results['hog'] = 'success'
        except Exception as e:
            results['hog'] = f'error: {e}'
            raise
        
        return results
    
    def get_available_models(self):
        """Returns dict of available models"""
        return self.available_models
    
    def is_model_loaded(self, model_id):
        """Check if a specific model is loaded"""
        return (model_id in self.available_models and 
                self.available_models[model_id].get('loaded', False))
