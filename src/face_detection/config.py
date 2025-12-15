import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(BASE_DIR, 'cascades', 'haarcascade_frontalface_default.xml')
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

EYE_CASCADE_PATH = os.path.join(BASE_DIR, 'cascades', 'haarcascade_eye.xml')
if not os.path.exists(EYE_CASCADE_PATH):
    EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'

DNN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'res10_300x300_ssd_iter_140000.caffemodel')
DNN_CONFIG_PATH = os.path.join(BASE_DIR, 'models', 'deploy.prototxt')
DNN_CONFIDENCE_THRESHOLD = 0.5

DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
ENGAGED_PATH = os.path.join(DATASET_PATH, 'Engaged')
NOT_ENGAGED_PATH = os.path.join(DATASET_PATH, 'Not engaged')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_model.h5')
SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')

HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (60, 60)
HAAR_MAX_SIZE = (500, 500)
HAAR_FLAGS = 0

ROI_PADDING_ADAPTIVE = True
ROI_PADDING_FIXED = 10
ROI_PADDING_PERCENT = 0.08

REJECT_LEVELS = True
LEVEL_WEIGHTS = True
FACE_QUALITY_THRESHOLD = 0.5

ROI_SIZE_EMOTION = (48, 48)
ROI_SIZE_ENGAGEMENT = (256, 256)
DEFAULT_ROI_SIZE = ROI_SIZE_ENGAGEMENT

APPLY_GRAYSCALE = True
APPLY_BLUR = True
APPLY_HISTOGRAM_EQ = True
APPLY_CLAHE = False
APPLY_DENOISING = False

BLUR_KERNEL_SIZE = (5, 5)
BLUR_SIGMA = 0

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

NORMALIZE_RANGE = (0, 1)
NORMALIZE_METHOD = 'minmax'

FACE_QUALITY_MIN_SHARPNESS = 0.3
FACE_QUALITY_MIN_BRIGHTNESS = 0.4
FACE_QUALITY_MIN_CONTRAST = 0.3
FACE_QUALITY_MIN_SIZE = 0.5

CLASS_LABELS = ['Engaged', 'Not Engaged']
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

ENGAGED_CATEGORIES = ['confused', 'engaged', 'frustrated']
NOT_ENGAGED_CATEGORIES = ['bored', 'drowsy', 'Looking Away']

WEBCAM_INDEX = 0
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
WEBCAM_FPS = 30

COLOR_ENGAGED = (0, 255, 0)
COLOR_NOT_ENGAGED = (0, 0, 255)
COLOR_BBOX = (255, 0, 0)
COLOR_TEXT = (255, 255, 255)

FONT_FACE = 0
FONT_SCALE = 0.7
FONT_THICKNESS = 2

SHOW_CONFIDENCE = True
SHOW_FPS = True
CONFIDENCE_THRESHOLD = 0.5

SKIP_FRAMES = 0
MAX_FACES = 5

ENABLE_SMOOTHING = True
SMOOTHING_WINDOW = 5
MIN_DETECTION_CONFIDENCE = 1
FACE_TRACKING_DISTANCE = 50
MAX_CONFIDENCE = 5
FACE_PERSISTENCE_FRAMES = 10

RESIZE_FOR_DETECTION = False
DETECTION_SCALE = 1.0