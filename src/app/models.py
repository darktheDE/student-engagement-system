import numpy as np
import cv2


class CNNModelWrapper:
    """Wrapper for CNN model that classifies 6 engagement states"""
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.class_names = ['bored', 'confused', 'drowsy', 'engaged', 'frustrated', 'looking away']
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            print(f"CNN model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            return False
    
    def preprocess(self, roi):
        """Preprocess face ROI for CNN input"""
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        
        roi = cv2.resize(roi, (256, 256))
        roi = roi.astype('float32')
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis=0)
        
        return roi
    
    def predict(self, roi):
        """
        Predict engagement state from face ROI
        Returns: (state, confidence)
        """
        if self.model is None:
            return None, 0.0
        
        try:
            processed = self.preprocess(roi)
            predictions = self.model.predict(processed, verbose=0)
            
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            state = self.class_names[class_idx]
            
            return state, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0


class SVMModelWrapper:
    def __init__(self, model_path=None):
        self.model = None
        self.hog = None
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)
        
        self._init_hog()
    
    def _init_hog(self):
        from skimage.feature import hog
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys'
        }
    
    def load_model(self, model_path):
        try:
            import joblib
            self.model = joblib.load(model_path)
            return True
        except Exception as e:
            return False
    
    def extract_hog_features(self, roi):
        from skimage.feature import hog
        
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        roi = cv2.resize(roi, (128, 128))
        
        features = hog(roi, **self.hog_params, feature_vector=True)
        return features
    
    def predict(self, roi):
        if self.model is None:
            return None, 0.0
        
        features = self.extract_hog_features(roi)
        features = features.reshape(1, -1)
        
        prediction = self.model.predict(features)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)
            confidence = np.max(proba)
        else:
            confidence = 1.0
        
        label = "Engaged" if prediction[0] == 1 else "Not Engaged"
        
        return label, confidence


class CombinedPredictor:
    def __init__(self, cnn_model, svm_model):
        self.cnn_model = cnn_model
        self.svm_model = svm_model
    
    def predict(self, roi, method='voting'):
        cnn_label, cnn_conf = self.cnn_model.predict(roi)
        svm_label, svm_conf = self.svm_model.predict(roi)
        
        if method == 'voting':
            return self._voting_predict(cnn_label, svm_label, cnn_conf, svm_conf)
        elif method == 'weighted':
            return self._weighted_predict(cnn_label, svm_label, cnn_conf, svm_conf)
        else:
            return self._confidence_based(cnn_label, svm_label, cnn_conf, svm_conf)
    
    def _voting_predict(self, cnn_label, svm_label, cnn_conf, svm_conf):
        if cnn_label == svm_label:
            return cnn_label, (cnn_conf + svm_conf) / 2
        
        if cnn_conf > svm_conf:
            return cnn_label, cnn_conf
        else:
            return svm_label, svm_conf
    
    def _weighted_predict(self, cnn_label, svm_label, cnn_conf, svm_conf):
        cnn_weight = 0.6
        svm_weight = 0.4
        
        cnn_score = cnn_conf if cnn_label == "Engaged" else (1 - cnn_conf)
        svm_score = svm_conf if svm_label == "Engaged" else (1 - svm_conf)
        
        final_score = cnn_weight * cnn_score + svm_weight * svm_score
        
        label = "Engaged" if final_score > 0.5 else "Not Engaged"
        return label, final_score
    
    def _confidence_based(self, cnn_label, svm_label, cnn_conf, svm_conf):
        if abs(cnn_conf - svm_conf) < 0.2:
            return self._voting_predict(cnn_label, svm_label, cnn_conf, svm_conf)
        
        if cnn_conf > svm_conf:
            return cnn_label, cnn_conf
        else:
            return svm_label, svm_conf
