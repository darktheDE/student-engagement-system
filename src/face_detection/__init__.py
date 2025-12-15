import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FaceDetector:
    
    def __init__(self, cascade_path=None, scale_factor=None, min_neighbors=None, 
                 min_size=None, max_size=None, use_dnn=False):
        self.cascade_path = cascade_path or config.CASCADE_PATH
        self.scale_factor = scale_factor or config.HAAR_SCALE_FACTOR
        self.min_neighbors = min_neighbors or config.HAAR_MIN_NEIGHBORS
        self.min_size = min_size or config.HAAR_MIN_SIZE
        self.max_size = max_size or config.HAAR_MAX_SIZE
        self.use_dnn = use_dnn
        
        self.face_cascade = self._load_cascade()
        self.profile_cascade = self._load_profile_cascade()
        self.eye_cascade = self._load_eye_cascade()
        
        self.dnn_net = self._load_dnn_model() if self.use_dnn else None
    
    def _load_profile_cascade(self):
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            return cascade if not cascade.empty() else None
        except:
            return None
    
    def _load_eye_cascade(self):
        try:
            path = getattr(config, 'EYE_CASCADE_PATH', None)
            if not path or not os.path.exists(path):
                path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            cascade = cv2.CascadeClassifier(path)
            return cascade if not cascade.empty() else None
        except:
            return None
    
    def _load_cascade(self):
        cascade_path = self.cascade_path
        if not os.path.exists(cascade_path):
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            opencv_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(opencv_path)
            if cascade.empty():
                raise ValueError(f"Failed to load Haar Cascade: {cascade_path}")
        return cascade
    
    def _load_dnn_model(self):
        try:
            model_path = getattr(config, 'DNN_MODEL_PATH', None)
            config_path = getattr(config, 'DNN_CONFIG_PATH', None)
            if model_path and config_path and os.path.exists(model_path):
                return cv2.dnn.readNetFromCaffe(config_path, model_path)
        except:
            pass
        return None
    
    def detect_faces(self, image, return_gray=False, method='haar'):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method == 'dnn' and self.use_dnn and self.dnn_net:
            faces = self._detect_faces_dnn(image)
        else:
            faces = self._detect_haar(gray_image)
        
        return (faces, gray_image) if return_gray else faces
    
    def _detect_haar(self, gray_image):
        faces = self.face_cascade.detectMultiScale(
            gray_image, self.scale_factor, self.min_neighbors,
            config.HAAR_FLAGS, self.min_size, self.max_size
        )
        
        if self.profile_cascade:
            profile_neighbors = max(self.min_neighbors + 2, 7)
            left_profiles = self.profile_cascade.detectMultiScale(
                gray_image, self.scale_factor, profile_neighbors,
                config.HAAR_FLAGS, self.min_size, self.max_size
            )
            
            flipped = cv2.flip(gray_image, 1)
            right_profiles = self.profile_cascade.detectMultiScale(
                flipped, self.scale_factor, profile_neighbors,
                config.HAAR_FLAGS, self.min_size, self.max_size
            )
            
            img_w = gray_image.shape[1]
            right_profiles = [(img_w - x - w, y, w, h) for x, y, w, h in right_profiles]
            
            all_faces = list(faces) if len(faces) > 0 else []
            all_faces.extend(left_profiles if len(left_profiles) > 0 else [])
            all_faces.extend(right_profiles if len(right_profiles) > 0 else [])
            faces = np.array(all_faces) if all_faces else np.array([])
        
        return faces
    
    def _detect_faces_dnn(self, image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        confidence_threshold = getattr(config, 'DNN_CONFIDENCE_THRESHOLD', 0.5)
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, xe, ye = box.astype("int")
                faces.append([x, y, xe - x, ye - y])
        
        return np.array(faces) if faces else np.array([])
    
    def extract_roi(self, image, face_coords, target_size=None, padding=0, 
                    adaptive_padding=True, validate_eyes=False):
        x, y, w, h = face_coords
        img_h, img_w = image.shape[:2]
        
        pad = int(max(w, h) * 0.08) if (adaptive_padding and padding == 0) else padding
        
        if pad > 0:
            x, y = max(0, x - pad), max(0, y - pad)
            w = min(img_w - x, w + 2 * pad)
            h = min(img_h - y, h + 2 * pad)
        
        roi = image[y:min(y + h, img_h), x:min(x + w, img_w)]
        
        if roi.size == 0:
            raise ValueError(f"Invalid ROI: ({x}, {y}, {w}, {h})")
        
        if validate_eyes and self.eye_cascade and not self._validate_roi_with_eyes(roi):
            pass
        
        if target_size:
            roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
        
        return roi
    
    def _validate_roi_with_eyes(self, roi):
        try:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            h, w = gray_roi.shape[:2]
            upper_face = gray_roi[0:int(h*0.6), :]
            eyes = self.eye_cascade.detectMultiScale(
                upper_face, scaleFactor=1.1, minNeighbors=3,
                minSize=(int(w*0.1), int(h*0.08))
            )
            return len(eyes) >= 2
        except:
            return True
    
    def preprocess_roi(self, roi, normalize=True, equalize_hist=False, 
                       apply_clahe=False, denoise=False):
        processed = roi.copy()
        
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        if denoise:
            processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
        
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        elif equalize_hist:
            processed = cv2.equalizeHist(processed)
        
        if normalize:
            processed = processed.astype('float32') / 255.0
        
        return processed
    
    def assess_face_quality(self, roi):
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        metrics = {}
        
        laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        metrics['sharpness'] = min(laplacian_var / 500.0, 1.0)
        
        mean_intensity = np.mean(gray_roi)
        metrics['brightness'] = 1.0 if 80 <= mean_intensity <= 170 else max(0, 1.0 - abs(mean_intensity - 125) / 125)
        
        metrics['contrast'] = min(np.std(gray_roi) / 60.0, 1.0)
        
        h, w = gray_roi.shape[:2]
        metrics['size'] = 1.0 if min(h, w) >= 100 else min(h, w) / 100
        
        weights = {'sharpness': 0.3, 'brightness': 0.25, 'contrast': 0.25, 'size': 0.2}
        quality_score = sum(metrics[k] * weights[k] for k in weights)
        
        return quality_score, metrics
    
    def detect_and_extract(self, image, target_size=None, preprocess=True, max_faces=None):
        faces = self.detect_faces(image)
        
        if max_faces is not None and len(faces) > max_faces:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max_faces]
        
        results = []
        for face_coords in faces:
            try:
                roi_raw = self.extract_roi(image, face_coords, target_size=target_size)
                
                if preprocess:
                    roi_processed = self.preprocess_roi(
                        roi_raw,
                        normalize=True,
                        equalize_hist=config.APPLY_HISTOGRAM_EQ
                    )
                else:
                    roi_processed = roi_raw
                
                results.append({
                    'bbox': tuple(face_coords),
                    'roi': roi_processed,
                    'roi_raw': roi_raw
                })
            except Exception:
                continue
        
        return results
    
    def draw_faces(self, image, faces, color=None, thickness=2, show_count=True):
        annotated = image.copy()
        color = color or config.COLOR_BBOX
        
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, thickness)
            
            if show_count:
                label = f"Face {i+1}"
                cv2.putText(
                    annotated, label, (x, y-10),
                    config.FONT_FACE, config.FONT_SCALE,
                    color, config.FONT_THICKNESS
                )
        
        return annotated


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(faces, iou_threshold=0.5):
    if len(faces) == 0:
        return []
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    keep = []
    while len(faces) > 0:
        keep.append(faces[0])
        faces = [f for f in faces[1:] if calculate_iou(faces[0], f) < iou_threshold]
    return keep



if __name__ == '__main__':
    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    dataset_path = config.ENGAGED_PATH
    if os.path.exists(dataset_path):
        confused_path = os.path.join(dataset_path, 'confused')
        if os.path.exists(confused_path):
            image_files = [f for f in os.listdir(confused_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                test_image_path = os.path.join(confused_path, image_files[0])
                image = cv2.imdecode(np.fromfile(test_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    faces = detector.detect_faces(image)
                    print(f"Detected {len(faces)} face(s)")
                    if len(faces) > 0:
                        results = detector.detect_and_extract(image, target_size=(256, 256))
                        print(f"Extracted {len(results)} ROI(s)")
