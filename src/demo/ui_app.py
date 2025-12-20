import cv2
import PIL.Image, PIL.ImageTk
import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import sys
import numpy as np
from collections import deque

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src.face_detection.face_detector import FaceDetector
from src.demo.utils import load_models, load_hog_model, custom_preprocess, draw_results, map_prediction_to_binary
from src.demo.hog_utils import extract_hog_features

# Constants
CNN_PATH = os.path.join(current_dir, 'models', 'CNN_feature.h5')
SVM_PATH = os.path.join(current_dir, 'models', 'svm_final_model.pkl')
HOG_SVM_PATH = os.path.join(current_dir, 'models', 'hog_svm_model.pkl')

WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
DISPLAY_WIDTH = 480
DISPLAY_HEIGHT = 360
FRAME_SKIP = 5
HISTORY_LEN = 100

class EngagementDemoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Engagement Classification Demo - Group 16")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        
        # --- Variables ---
        self.running = True
        self.system_ready = False
        self.cap = None
        self.frame_count = 0
        
        # Detectors & Models
        self.detector = None
        self.cnn_model = None
        self.svm_model = None
        self.hog_model = None
        
        # Data
        self.current_faces = []
        self.predictions_cnn = []
        self.predictions_hog = [] 
        
        # Metrics History
        self.history_cnn = deque(maxlen=HISTORY_LEN)
        self.history_hog = deque(maxlen=HISTORY_LEN)
        self.history_cnn_raw = deque(maxlen=HISTORY_LEN)
        self.history_hog_raw = deque(maxlen=HISTORY_LEN)
        
        # FPS
        self.fps_start = time.time()
        self.fps_counter = 0
        self.fps_val = 0

        # --- UI Layout ---
        self._build_header()
        self._build_main_content()
        self._build_footer()

        # --- Initialization ---
        self.log("Initializing models...")
        # Run init in thread to not freeze UI
        threading.Thread(target=self._initialize_system, daemon=True).start()
        # Start loop on main thread, waiting for system_ready
        self._update_loop()

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=20)
        header_frame.pack(fill=tk.X)
        
        title = tk.Label(header_frame, 
                         text="PHÂN LOẠI MỨC ĐỘ HỨNG THÚ HỌC TẬP CỦA SINH VIÊN\n BẰNG PHÂN TÍCH KHUÔN MẶT",
                         font=("Arial", 20, "bold"), fg="white", bg="#2c3e50", justify=tk.CENTER)
        title.pack()
        
        group = tk.Label(header_frame, text="Nhóm 16", 
                         font=("Arial", 16, "bold"), fg="#f1c40f", bg="#2c3e50")
        group.pack(pady=5)
        
        members = "Huỳnh Ngọc Thạch - 23133072  |  Huỳnh Hữu Huy - 23133027  |  Đỗ Kiến Hưng - 23133030  |  Nguyễn Tấn Thành - 23133068"
        mem_lbl = tk.Label(header_frame, text=members, 
                           font=("Arial", 12), fg="#ecf0f1", bg="#2c3e50")
        mem_lbl.pack()

    def _build_main_content(self):
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # --- LEFT: VIDEO PANELS ---
        video_container = tk.Frame(main_frame, bg="#f0f0f0")
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Panel 1: CNN
        self.panel_cnn = self._create_video_panel(video_container, "CNN + SVM Model", "#3498db")
        self.panel_cnn.pack(side=tk.LEFT, padx=10)
        
        # Panel 2: HOG
        self.panel_hog = self._create_video_panel(video_container, "HOG + SVM Model", "#e67e22")
        self.panel_hog.pack(side=tk.LEFT, padx=10)
        
        # --- RIGHT: SIDEBAR INFO ---
        sidebar = tk.Frame(main_frame, bg="white", width=300, relief=tk.RIDGE, bd=2)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        sidebar.pack_propagate(False) # Enforce width
        
        tk.Label(sidebar, text="THỐNG KÊ", font=("Arial", 16, "bold"), bg="white", fg="#2c3e50").pack(pady=20)
        
        # FPS
        self.lbl_fps = self._create_metric(sidebar, "FPS Real-time", "0")
        
        # Face Count
        self.lbl_faces = self._create_metric(sidebar, "Khuôn mặt phát hiện", "0")
        
        tk.Frame(sidebar, height=2, bg="#bdc3c7").pack(fill=tk.X, padx=20, pady=20)
        
        # Engagement CNN
        self.lbl_eng_cnn = self._create_metric(sidebar, "Độ hứng thú (CNN)", "0.0%", "#3498db")
        self.lbl_detail_cnn = tk.Label(sidebar, text="", font=("Arial", 10, "italic"), bg="white", fg="#7f8c8d")
        self.lbl_detail_cnn.pack(anchor="w", padx=20)
        
        # Engagement HOG
        self.lbl_eng_hog = self._create_metric(sidebar, "Độ hứng thú (HOG)", "0.0%", "#e67e22")
        self.lbl_detail_hog = tk.Label(sidebar, text="", font=("Arial", 10, "italic"), bg="white", fg="#7f8c8d")
        self.lbl_detail_hog.pack(anchor="w", padx=20)

    def _create_video_panel(self, parent, title, color):
        frame = tk.Frame(parent, bg="white", bd=2, relief=tk.GROOVE)
        
        lbl_title = tk.Label(frame, text=title, font=("Arial", 14, "bold"), bg=color, fg="white", pady=5)
        lbl_title.pack(fill=tk.X)
        
        canvas = tk.Label(frame, bg="black", width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
        canvas.pack(padx=5, pady=5)
        
        frame.canvas = canvas
        return frame

    def _create_metric(self, parent, title, value_text, value_color="black"):
        frame = tk.Frame(parent, bg="white")
        frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(frame, text=title, font=("Arial", 10, "bold"), bg="white", fg="gray").pack(anchor="w")
        val_lbl = tk.Label(frame, text=value_text, font=("Arial", 24, "bold"), bg="white", fg=value_color)
        val_lbl.pack(anchor="w")
        return val_lbl

    def _build_footer(self):
        footer = tk.Frame(self.root, bg="#f0f0f0", pady=10)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_lbl = tk.Label(footer, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_lbl.pack(fill=tk.X)

    def log(self, text):
        print(text)
        if hasattr(self, 'status_lbl'):
            self.status_lbl.config(text=text)

    # --- LOGIC ---
    def _initialize_system(self):
        try:
            # Load Models
            self.cnn_model, self.svm_model = load_models(CNN_PATH, SVM_PATH)
            self.log("CNN + SVM Loaded.")
            
            self.hog_model = load_hog_model(HOG_SVM_PATH)
            self.log("HOG + SVM Loaded.")
            
            # Init Detector (Optimized)
            self.detector = FaceDetector(use_dnn=False, scale_factor=1.2, min_neighbors=5)
            self.log("Face Detector Initialized.")
            
            # Start Camera
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                self.log("ERROR: Webcam not found.")
                return

            self.log("System Running...")
            self.system_ready = True
            
        except Exception as e:
            self.log(f"Error Init: {e}")

    def _update_loop(self):
        if not self.running:
            return

        if not self.system_ready:
            # Wait for init to finish
            self.root.after(100, self._update_loop)
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Mirror frame
            frame = cv2.flip(frame, 1)
            frame_resized = cv2.resize(frame, (WEBCAM_WIDTH, WEBCAM_HEIGHT))
            
            # Processing Logic (Threaded ideally, but keeping simple here for Tkinter compatibility specific mainly to image conversion)
            # To maintain responsiveness, we do heavy lifting here but keep it efficient.
            
            if self.frame_count % FRAME_SKIP == 0:
                self._process_frame(frame_resized)
            
            # Visualization
            img_cnn = self._visualize(frame_resized.copy(), self.predictions_cnn, "CNN")
            img_hog = self._visualize(frame_resized.copy(), self.predictions_hog, "HOG")
            
            # Update UI
            self._update_image(self.panel_cnn.canvas, img_cnn)
            self._update_image(self.panel_hog.canvas, img_hog)
            
            # Update Metrics
            self._update_metrics()
            
            # FPS Calculation
            self.fps_counter += 1
            if time.time() - self.fps_start >= 1.0:
                self.fps_val = self.fps_counter
                self.fps_counter = 0
                self.fps_start = time.time()
                
            self.frame_count += 1
        
        # Schedule next frame (approx 30ms for ~30fps)
        self.root.after(30, self._update_loop)

    def _process_frame(self, frame):
        self.current_faces = self.detector.detect_faces(frame)
        self.predictions_cnn = []
        self.predictions_hog = []
        
        for face_rect in self.current_faces:
            try:
                roi = self.detector.extract_roi(frame, face_rect, adaptive_padding=True)
                if roi.size == 0:
                    self.predictions_cnn.append(-1)
                    self.predictions_hog.append(-1)
                    continue
                
                # --- CNN Pipeline ---
                input_blob = custom_preprocess(roi)
                features_cnn = self.cnn_model.predict(input_blob, verbose=0)
                if len(features_cnn.shape) > 2: features_cnn = features_cnn.reshape(1, -1)
                pred_cnn = self.svm_model.predict(features_cnn)[0]
                self.predictions_cnn.append(pred_cnn)
                # Map 0-5 prediction to 0-1 for metrics
                self.history_cnn.append(map_prediction_to_binary(pred_cnn))
                self.history_cnn_raw.append(pred_cnn)

                # --- HOG Pipeline ---
                features_hog = extract_hog_features(roi)
                features_hog = features_hog.reshape(1, -1)
                pred_hog = self.hog_model.predict(features_hog)[0]
                self.predictions_hog.append(pred_hog)
                # Map 0-5 prediction to 0-1 for metrics
                self.history_hog.append(map_prediction_to_binary(pred_hog))
                self.history_hog_raw.append(pred_hog)

            except Exception as e:
                print(f"Error processing: {e}")
                self.predictions_cnn.append(-1)
                self.predictions_hog.append(-1)

    def _visualize(self, img, predictions, model_type="CNN"):
        for i, face_rect in enumerate(self.current_faces):
            if i < len(predictions):
                pred = predictions[i]
                if pred != -1:
                    img = draw_results(img, face_rect, pred)
        
        # Resize for Display Panel
        img = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        return img

    def _update_image(self, canvas, opencv_img):
        # Convert BGR to RGB
        color = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(color)
        photo = PIL.ImageTk.PhotoImage(image)
        
        canvas.configure(image=photo)
        canvas.image = photo

    def _update_metrics(self):
        self.lbl_fps.config(text=f"{self.fps_val}")
        self.lbl_faces.config(text=f"{len(self.current_faces)}")
        
        if len(self.history_cnn) > 0:
            rate_cnn = (sum(self.history_cnn) / len(self.history_cnn)) * 100
            self.lbl_eng_cnn.config(text=f"{rate_cnn:.1f}%", fg="#2ecc71" if rate_cnn > 50 else "#e74c3c")
            
            # Breakdown
            from src.demo.stats_utils import calculate_state_breakdown
            from src.demo.utils import LABEL_MAP
            breakdown_cnn = calculate_state_breakdown(self.history_cnn_raw, LABEL_MAP)
            self.lbl_detail_cnn.config(text=breakdown_cnn)
        
        if len(self.history_hog) > 0:
            rate_hog = (sum(self.history_hog) / len(self.history_hog)) * 100
            self.lbl_eng_hog.config(text=f"{rate_hog:.1f}%", fg="#2ecc71" if rate_hog > 50 else "#e74c3c")
            
            # Breakdown
            from src.demo.stats_utils import calculate_state_breakdown
            from src.demo.utils import LABEL_MAP
            breakdown_hog = calculate_state_breakdown(self.history_hog_raw, LABEL_MAP)
            self.lbl_detail_hog.config(text=breakdown_hog)

    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EngagementDemoUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
