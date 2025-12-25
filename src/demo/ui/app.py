"""
Main Application - Student Engagement Classification Demo
Coordinates all components and manages application lifecycle
"""
import tkinter as tk
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
import sys
import os

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
demo_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(demo_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.face_detection.face_detector import FaceDetector
from src.demo.utils.preprocessing import preprocess_image

from src.demo.core import ModelManager, Predictor, VideoProcessor
from src.demo.config import (DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_WIDTH_SPLIT, 
                      DISPLAY_HEIGHT_SPLIT, FACE_DETECTION_SCALE_FACTOR,
                      FACE_DETECTION_MIN_NEIGHBORS, LABEL_MAP, COLOR_BACKGROUND)
from src.demo.utils import (calculate_engagement_rate, calculate_state_breakdown,
                         calculate_agreement_rate, calculate_confidence,
                         map_prediction_to_binary)
from src.demo.utils.visualization import draw_prediction_on_frame, draw_clean_rectangle
from src.demo.utils.image_utils import convert_cv2_to_tkinter

from src.demo.ui.components import (HeaderComponent, SidebarComponent, VideoPanelComponent,
                         MetricsPanelComponent, FooterComponent)


class EngagementApp:
    """Main application class"""
    
    def __init__(self, root):
        """
        Initialize application
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Student Engagement Classification Demo - Group 16")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLOR_BACKGROUND)
        
        # Application state
        self.running = True
        self.system_ready = False
        
        # Threading for performance
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        
        # Core components
        self.model_manager = None
        self.predictor = None
        self.video_processor = None
        self.detector = None
        
        # UI components
        self.header = None
        self.sidebar = None
        self.video_panel = None
        self.metrics_panel = None
        self.footer = None
        
        # Build UI
        self._build_ui()
        
        # Initialize system in background
        self.log("Initializing models...")
        threading.Thread(target=self._initialize_system, daemon=True).start()
        
        # Start update loop
        self._update_loop()
    
    def _build_ui(self):
        """Build all UI components"""
        # Header
        self.header = HeaderComponent(self.root)
        
        # Main content area
        main_frame = tk.Frame(self.root, bg=COLOR_BACKGROUND)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Video panel (left)
        self.video_panel = VideoPanelComponent(main_frame)
        self.video_panel.set_mode_change_callback(self._on_mode_change)
        
        # Sidebar (right)
        self.sidebar = SidebarComponent(main_frame)
        
        # Metrics panel (inside sidebar)
        self.metrics_panel = MetricsPanelComponent(self.sidebar.get_metrics_container())
        self.metrics_panel.build_single_metrics()  # Default view
        
        # Footer
        self.footer = FooterComponent(self.root)
    
    def _initialize_system(self):
        """Initialize all system components in background"""
        try:
            # Initialize model manager
            self.log("Loading models...")
            self.model_manager = ModelManager()
            results = self.model_manager.load_all_models()
            
            for model_id, status in results.items():
                if 'success' in status:
                    self.log(f"✓ {model_id} loaded")
                elif 'not_found' in status:
                    self.log(f"⚠ {model_id} not found")
                else:
                    self.log(f"❌ {model_id} failed: {status}")
            
            # Initialize predictor
            self.predictor = Predictor(self.model_manager)
            self.log("✓ Predictor initialized")
            
            # Initialize face detector
            self.detector = FaceDetector(
                use_dnn=False,
                scale_factor=FACE_DETECTION_SCALE_FACTOR,
                min_neighbors=FACE_DETECTION_MIN_NEIGHBORS
            )
            self.log("✓ Face Detector initialized")
            
            # Initialize video processor
            self.video_processor = VideoProcessor(self.detector)
            
            if not self.video_processor.initialize_camera():
                self.log("❌ Failed to initialize camera")
                return
            
            # Initialize history for all models
            model_ids = list(self.model_manager.get_available_models().keys())
            self.video_processor.initialize_history(model_ids)
            
            self.log("🎥 System Running...")
            self.system_ready = True
            
        except Exception as e:
            self.log(f"❌ Error Init: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_loop(self):
        """Main update loop"""
        if not self.running:
            return
        
        if not self.system_ready:
            self.root.after(100, self._update_loop)
            return
        
        # Read frame (Fast)
        frame = self.video_processor.read_frame()
        
        if frame is not None:
            # Apply brightness adjustment from UI (Tkinter access on main thread)
            brightness = self.sidebar.brightness_adjust.get()
            self.video_processor.set_brightness(brightness)
            
            # Check async processing status
            if self.future and self.future.done():
                try:
                    # Get results from background thread (updates state silently via side effects or we could return them)
                    self.future.result() 
                    self.future = None
                except Exception as e:
                    print(f"Error in background processing: {e}")
                    self.future = None
            
            # Submit new processing task if idle
            if self.future is None and self.video_processor.should_process_frame():
                # Gather Tkinter inputs here (Main Thread)
                mode = self.video_panel.view_mode.get()
                models_to_run = self._get_models_to_run(mode)
                
                # Submit to background thread (Do NOT pass Tkinter objects)
                self.future = self.executor.submit(self._process_frame_async, frame.copy(), models_to_run)
            
            # Visualize based on mode (Uses current state of video_processor)
            self._visualize_frame(frame)
            
            # Update metrics
            self._update_metrics()
            
            # Update FPS
            self.video_processor.update_fps()
        
        # Schedule next update
        self.root.after(30, self._update_loop)
    
    def _process_frame_async(self, frame, models_to_run):
        """
        Process a frame - detect faces and make predictions (Background Thread).
        WARNING: Do NOT access Tkinter widgets here.
        """
        # Detect faces
        faces = self.video_processor.detect_faces(frame)
        
        if len(faces) == 0:
            return
        
        # Get largest face
        face_rect = self.video_processor.get_largest_face()
        if face_rect is None:
            return
        
        # Extract ROI
        roi = self.video_processor.extract_face_roi(frame, face_rect)
        if roi is None:
            return
        
        # Run predictions
        for model_id in models_to_run:
            if not self.model_manager.is_model_loaded(model_id):
                continue
            
            model_start = time.time()
            pred = self.predictor.predict_with_model(roi, model_id)
            model_time = (time.time() - model_start) * 1000
            
            if pred is not None:
                # Use update_with_smoothing to handle buffer and history
                self.video_processor.update_with_smoothing(model_id, pred, map_prediction_to_binary)
                self.video_processor.processing_times[model_id] = model_time

    
    def _get_models_to_run(self, mode):
        """Determine which models to run based on view mode"""
        models_to_run = set()
        
        if mode == "single":
            models_to_run.add(self.video_panel.model_left.get())
        elif mode == "comparison":
            models_to_run.add(self.video_panel.model_left.get())
            models_to_run.add(self.video_panel.model_right.get())
        elif mode == "stats":
            models_to_run = set(self.model_manager.get_available_models().keys())
        
        return models_to_run
    
    def _visualize_frame(self, frame):
        """Visualize frame based on current mode"""
        mode = self.video_panel.view_mode.get()
        canvases = self.video_panel.get_canvases()
        
        if mode == "single":
            model_id = self.video_panel.model_left.get()
            img = self._create_visualization(frame.copy(), model_id)
            self._update_canvas(canvases['single'], img, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        
        elif mode == "comparison":
            model_left = self.video_panel.model_left.get()
            model_right = self.video_panel.model_right.get()
            
            img_left = self._create_visualization(frame.copy(), model_left)
            img_right = self._create_visualization(frame.copy(), model_right)
            
            self._update_canvas(canvases['left'], img_left, DISPLAY_WIDTH_SPLIT, DISPLAY_HEIGHT_SPLIT)
            self._update_canvas(canvases['right'], img_right, DISPLAY_WIDTH_SPLIT, DISPLAY_HEIGHT_SPLIT)
        
        elif mode == "stats":
            img = self._create_clean_visualization(frame.copy())
            self._update_canvas(canvases['single'], img, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    def _create_visualization(self, img, model_id):
        """Create visualization for a specific model"""
        if len(self.video_processor.current_faces) == 0:
            return img
        
        face_rect = self.video_processor.get_largest_face()
        if face_rect is None:
            return img
        
        # Show preprocessed image if toggled
        if self.video_panel.show_preprocessed.get():
            try:
                roi = self.video_processor.extract_face_roi(img, face_rect)
                if roi is not None and roi.size > 0:
                    roi_processed = preprocess_image(roi, target_size=128)
                    x, y, w, h = face_rect
                    roi_processed_resized = cv2.resize(roi_processed, (w, h))
                    roi_bgr = cv2.cvtColor(roi_processed_resized, cv2.COLOR_GRAY2BGR)
                    img[y:y+h, x:x+w] = roi_bgr
            except Exception as e:
                print(f"Error showing preprocessed: {e}")
        
        # Draw prediction if available
        pred = self.video_processor.get_smoothed_prediction(model_id)
        if pred is not None:
            img = draw_prediction_on_frame(img, face_rect, pred)
        else:
            img = draw_clean_rectangle(img, face_rect)
        
        return img
    
    def _create_clean_visualization(self, img):
        """Create visualization without predictions"""
        if len(self.video_processor.current_faces) > 0:
            face_rect = self.video_processor.get_largest_face()
            if face_rect is not None:
                img = draw_clean_rectangle(img, face_rect)
        return img
    
    def _update_canvas(self, canvas, opencv_img, width, height):
        """Update a canvas with an image"""
        photo = convert_cv2_to_tkinter(opencv_img, width, height)
        canvas.configure(image=photo)
        canvas.image = photo  # Keep a reference
    
    def _update_metrics(self):
        """Update all metrics displays"""
        # Common metrics
        self.sidebar.lbl_fps.config(text=f"{self.video_processor.get_fps()}")
        self.sidebar.lbl_faces.config(text=f"{len(self.video_processor.current_faces)}")
        self.sidebar.lbl_light_quality.config(text=self.video_processor.calculate_light_quality())
        
        # Processing time
        if self.video_processor.processing_times:
            avg_time = sum(self.video_processor.processing_times.values()) / len(self.video_processor.processing_times)
            self.sidebar.lbl_process_time.config(text=f"{avg_time:.0f} ms")
        
        # Mode-specific metrics
        mode = self.video_panel.view_mode.get()
        
        if mode == "single":
            self._update_single_metrics()
        elif mode == "comparison":
            self._update_comparison_metrics()
        elif mode == "stats":
            self._update_stats_metrics()
    
    def _update_single_metrics(self):
        """Update metrics for single model view"""
        model_id = self.video_panel.model_left.get()
        
        if model_id not in self.video_processor.history:
            return
        
        history = self.video_processor.history[model_id]
        
        if len(history['binary']) == 0:
            return
        
        # Engagement rate
        rate = calculate_engagement_rate(history['binary'])
        color = "#2ecc71" if rate > 50 else "#e74c3c"
        self.metrics_panel.lbl_engagement.config(text=f"{rate:.1f}%", fg=color)
        self.metrics_panel.progress_var.set(rate)
        
        # Breakdown
        breakdown = calculate_state_breakdown(history['raw'], LABEL_MAP)
        self.metrics_panel.lbl_breakdown.config(text=breakdown)
        
        # Confidence
        confidence = calculate_confidence(history['raw'], window_size=10)
        self.metrics_panel.lbl_confidence.config(text=f"{confidence:.0f}%")
    
    def _update_comparison_metrics(self):
        """Update metrics for comparison view"""
        model_left = self.video_panel.model_left.get()
        model_right = self.video_panel.model_right.get()
        
        # Left model
        if model_left in self.video_processor.history:
            history = self.video_processor.history[model_left]
            if len(history['binary']) > 0:
                rate = calculate_engagement_rate(history['binary'])
                self.metrics_panel.lbl_left_engagement.config(text=f"{rate:.1f}%")
                
                breakdown = calculate_state_breakdown(history['raw'], LABEL_MAP)
                self.metrics_panel.lbl_left_breakdown.config(text=breakdown)
        
        # Right model
        if model_right in self.video_processor.history:
            history = self.video_processor.history[model_right]
            if len(history['binary']) > 0:
                rate = calculate_engagement_rate(history['binary'])
                self.metrics_panel.lbl_right_engagement.config(text=f"{rate:.1f}%")
                
                breakdown = calculate_state_breakdown(history['raw'], LABEL_MAP)
                self.metrics_panel.lbl_right_breakdown.config(text=breakdown)
        
        # Agreement
        if (model_left in self.video_processor.history and 
            model_right in self.video_processor.history):
            agreement = calculate_agreement_rate(
                self.video_processor.history[model_left]['raw'],
                self.video_processor.history[model_right]['raw']
            )
            self.metrics_panel.lbl_agreement.config(text=f"{agreement:.1f}%")
    
    def _update_stats_metrics(self):
        """Update metrics for stats view"""
        for model_id in self.metrics_panel.stats_labels:
            if model_id in self.video_processor.history:
                history = self.video_processor.history[model_id]
                if len(history['binary']) > 0:
                    rate = calculate_engagement_rate(history['binary'])
                    self.metrics_panel.stats_labels[model_id]['engagement'].config(text=f"{rate:.1f}%")
            
            if model_id in self.video_processor.processing_times:
                time_ms = self.video_processor.processing_times[model_id]
                self.metrics_panel.stats_labels[model_id]['time'].config(text=f"{time_ms:.0f}ms")
    
    def _on_mode_change(self):
        """Handle view mode change"""
        mode = self.video_panel.view_mode.get()
        self.log(f"Chế độ: {mode}")
        
        # Rebuild metrics panel
        if mode == "single":
            self.metrics_panel.build_single_metrics()
        elif mode == "comparison":
            self.metrics_panel.build_comparison_metrics()
        elif mode == "stats":
            self.metrics_panel.build_stats_metrics()
    
    def log(self, text):
        """Log message to console and status bar"""
        print(text)
        if self.footer:
            self.footer.update_status(text)
    
    def on_closing(self):
        """Handle application closing"""
        self.running = False
        if self.video_processor:
            self.video_processor.release()
        self.root.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = EngagementApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"Error starting app: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
