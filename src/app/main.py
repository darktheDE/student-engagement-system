import sys
import os
import cv2
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QTextEdit, QGroupBox)
from PyQt6.QtCore import Qt, QTimer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from face_detection import FaceDetector

try:
    from .ui_components import create_header
    from .styles import MAIN_WINDOW_STYLE
    from .utils import cv_to_pixmap, draw_face_boxes, generate_metrics_html
    from .chart_utils import generate_timeline_chart_html
    from .engagement_config import ENGAGEMENT_STATES, get_state_info, calculate_engagement_score
    from .models import CNNModelWrapper
except ImportError:
    from ui_components import create_header
    from styles import MAIN_WINDOW_STYLE
    from utils import cv_to_pixmap, draw_face_boxes, generate_metrics_html
    from chart_utils import generate_timeline_chart_html
    from engagement_config import ENGAGEMENT_STATES, get_state_info, calculate_engagement_score
    from models import CNNModelWrapper


class StudentEngagementApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phân Loại Mức Độ Hứng Thú Học Tập Của Sinh Viên")
        self.setGeometry(50, 50, 1600, 900)
        self.showMaximized()
        
        self.detector = FaceDetector()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.total_faces_detected = 0
        self.frame_count = 0
        
        # Model and state tracking
        self.cnn_model = None
        self.engaged_count = 0
        self.not_engaged_count = 0
        self.state_counts = {'engaged': 0, 'confused': 0, 'frustrated': 0, 'bored': 0, 'drowsy': 0, 'looking away': 0}
        self.current_state = None
        self.current_confidence = 0.0
        self.engagement_timeline = []
        self.last_predictions = []  # Cache predictions to reduce lag
        
        self.start_time = datetime.now()
        
        self.init_ui()
        self.apply_styles()
        self.load_models()
        self.start_webcam()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        header = create_header()
        main_layout.addWidget(header)
        
        content_layout = QHBoxLayout()
        content_layout.setSpacing(8)
        
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()
        
        content_layout.addLayout(left_panel, 70)
        content_layout.addLayout(right_panel, 30)
        
        main_layout.addLayout(content_layout, 1)
    
    
    def log_event(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.append(f"[{timestamp}] {message}")
    
    def create_left_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel("Đang khởi động camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setObjectName("videoLabel")
        layout.addWidget(self.video_label, 1)
        
        return layout
    
    def create_right_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # A. Kết Quả Phân Loại (Real-time)
        result_group = QGroupBox("Kết Quả Phân Loại")
        result_group.setObjectName("resultGroup")
        result_layout = QVBoxLayout(result_group)
        result_layout.setSpacing(6)
        result_layout.setContentsMargins(8, 12, 8, 8)
        
        # Trạng thái tổng quát (TO, RÕ RÀNG)
        self.overall_status_label = QLabel("ĐANG TẬP TRUNG")
        self.overall_status_label.setObjectName("overallStatusLabel")
        self.overall_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overall_status_label.setMinimumHeight(50)
        self.overall_status_label.setWordWrap(True)
        result_layout.addWidget(self.overall_status_label)
        
        # Cảm xúc/Trạng thái chi tiết
        self.detail_status_label = QLabel("Hứng thú")
        self.detail_status_label.setObjectName("detailStatusLabel")
        self.detail_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_status_label.setMinimumHeight(35)
        result_layout.addWidget(self.detail_status_label)
        
        # Thanh mức độ tin cậy (Confidence Bar)
        confidence_container = QWidget()
        confidence_layout = QVBoxLayout(confidence_container)
        confidence_layout.setContentsMargins(0, 5, 0, 0)
        confidence_layout.setSpacing(3)
        
        self.confidence_label = QLabel("Độ tin cậy: 0%")
        self.confidence_label.setObjectName("confidenceLabel")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        confidence_layout.addWidget(self.confidence_label)
        
        # Progress bar (dùng QTextEdit để hiển thị HTML progress bar)
        self.confidence_bar = QTextEdit()
        self.confidence_bar.setObjectName("confidenceBar")
        self.confidence_bar.setReadOnly(True)
        self.confidence_bar.setMaximumHeight(30)
        self.confidence_bar.setMinimumHeight(30)
        self.confidence_bar.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.confidence_bar.setHtml(self._generate_confidence_html(0))
        confidence_layout.addWidget(self.confidence_bar)
        
        result_layout.addWidget(confidence_container)
        layout.addWidget(result_group)
        
        # B. Thống Kê Chi Tiết (Statistics)
        metrics_group = QGroupBox("Thống Kê Chi Tiết")
        metrics_group.setObjectName("metricsGroup")
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setContentsMargins(8, 12, 8, 8)
        self.metrics_text = QTextEdit()
        self.metrics_text.setObjectName("metricsText")
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumHeight(120)
        metrics_layout.addWidget(self.metrics_text)
        layout.addWidget(metrics_group, 1)
        
        # C. Thông Tin Phiên (Session Log / Timeline)
        info_group = QGroupBox("Thông Tin Phiên")
        info_group.setObjectName("infoGroup")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(8, 12, 8, 8)
        info_layout.setSpacing(5)
        
        # Timeline chart
        self.timeline_chart = QTextEdit()
        self.timeline_chart.setObjectName("timelineChart")
        self.timeline_chart.setReadOnly(True)
        self.timeline_chart.setMinimumHeight(120)
        self.timeline_chart.setMaximumHeight(140)
        self.timeline_chart.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        info_layout.addWidget(self.timeline_chart)
        
        # Event log
        self.info_text = QTextEdit()
        self.info_text.setObjectName("infoText")
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(60)
        info_layout.addWidget(self.info_text, 1)
        layout.addWidget(info_group, 1)
        
        return layout
    
    def _generate_confidence_html(self, confidence):
        pct = int(confidence * 100)
        color = "#10b981" if confidence >= 0.7 else "#f59e0b" if confidence >= 0.5 else "#ef4444"
        return f"""
        <div style='padding: 0; margin: 0;'>
            <div style='width: 100%; height: 20px; background-color: #1a1f2e; border-radius: 10px; overflow: hidden; border: 2px solid #374151;'>
                <div style='width: {pct}%; height: 100%; background: linear-gradient(90deg, {color} 0%, {color} 100%); transition: width 0.3s;'></div>
            </div>
        </div>
        """
    
    def load_models(self):
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'Student Engagement Model.h5')
            if os.path.exists(model_path):
                self.cnn_model = CNNModelWrapper(model_path)
                self.log_event("CNN model loaded successfully")
            else:
                self.log_event(f"Model not found: {model_path}")
            
            self.timeline_chart.setHtml(generate_timeline_chart_html([]))
        except Exception as e:
            self.log_event(f"Error loading model: {str(e)}")
    
    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.timer.start(30)
            self.log_event("Camera khởi động thành công")
            self.start_time = datetime.now()
            self.total_faces_detected = 0
            self.frame_count = 0
            self.engaged_count = 0
            self.not_engaged_count = 0
            self.state_counts = {k: 0 for k in self.state_counts}
            self.engagement_timeline = []
            self.update_metrics_display()
        else:
            self.log_event("LỖI: Không thể mở camera")
            self.video_label.setText("Không thể khởi động camera")
    
    def stop_webcam(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.setText("Camera đã dừng")
        duration = (datetime.now() - self.start_time).total_seconds()
        self.log_event(f"Camera dừng. Thời gian: {duration:.0f}s")
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.frame_count += 1
        faces = self.detector.detect_faces(frame)
        self.total_faces_detected += len(faces)
        
        # Dự đoán engagement cho từng face (chỉ mỗi 3 frames để giảm lag)
        predictions = []
        if self.frame_count % 3 == 0:  # Chỉ predict mỗi 3 frames
            for face_bbox in faces:
                try:
                    roi = self.detector.extract_roi(frame, face_bbox, target_size=(256, 256))
                    state, confidence = self.predict_engagement(roi)
                    predictions.append({'state': state, 'confidence': confidence})
                    
                    # Cập nhật state counts
                    if state:
                        self.state_counts[state] = self.state_counts.get(state, 0) + 1
                        state_info = get_state_info(state)
                        if state_info['group'] == 'Engaged':
                            self.engaged_count += 1
                        else:
                            self.not_engaged_count += 1
                except Exception as e:
                    predictions.append({'state': None, 'confidence': 0.0})
        else:
            # Sử dụng prediction cũ nếu có
            if hasattr(self, 'last_predictions') and self.last_predictions:
                predictions = self.last_predictions
            else:
                predictions = [{'state': None, 'confidence': 0.0} for _ in faces]
        
        # Lưu predictions để dùng cho frames tiếp theo
        if predictions and predictions[0]['state']:
            self.last_predictions = predictions
        
        # Vẽ bounding boxes với predictions
        frame, rois_info = draw_face_boxes(frame, faces, self.detector, predictions)
        
        pixmap = cv_to_pixmap(frame, self.video_label.size())
        self.video_label.setPixmap(pixmap)
        
        # Cập nhật trạng thái hiện tại (lấy trạng thái phổ biến nhất)
        if predictions and predictions[0]['state']:
            self.current_state = predictions[0]['state']
            self.current_confidence = predictions[0]['confidence']
            state_info = get_state_info(self.current_state)
            
            # Trạng thái tổng quát (TO, RÕ RÀNG)
            if state_info['group'] == 'Engaged':
                self.overall_status_label.setText("ĐANG TẬP TRUNG")
                self.overall_status_label.setStyleSheet("""
                    QLabel#overallStatusLabel {
                        color: #10b981;
                        font-size: 20px;
                        font-weight: bold;
                        background-color: #1a1f2e;
                        border-radius: 8px;
                        padding: 10px;
                    }
                """)
            else:
                self.overall_status_label.setText("KHÔNG TẬP TRUNG")
                self.overall_status_label.setStyleSheet("""
                    QLabel#overallStatusLabel {
                        color: #ef4444;
                        font-size: 20px;
                        font-weight: bold;
                        background-color: #1a1f2e;
                        border-radius: 8px;
                        padding: 10px;
                    }
                """)
            
            # Cảm xúc/Trạng thái chi tiết
            self.detail_status_label.setText(state_info['display_name'])
            self.detail_status_label.setStyleSheet(f"""
                QLabel#detailStatusLabel {{
                    color: {state_info['color_hex']};
                    font-size: 16px;
                    font-weight: 600;
                    background-color: #1a1f2e;
                    border-radius: 6px;
                    padding: 8px;
                }}
            """)
            
            # Thanh confidence
            self.confidence_label.setText(f"Độ tin cậy: {self.current_confidence:.0%}")
            self.confidence_bar.setHtml(self._generate_confidence_html(self.current_confidence))
            
        elif len(faces) > 0:
            self.overall_status_label.setText("Đang phân tích...")
            self.detail_status_label.setText("Chờ giây lát...")
            self.confidence_label.setText("Độ tin cậy: 0%")
            self.confidence_bar.setHtml(self._generate_confidence_html(0))
        else:
            self.overall_status_label.setText("Chưa phát hiện")
            self.detail_status_label.setText("---")
            self.confidence_label.setText("Độ tin cậy: 0%")
            self.confidence_bar.setHtml(self._generate_confidence_html(0))
        
        # Log định kỳ và cập nhật timeline chart
        if self.frame_count % 300 == 0:  # Log mỗi 10s
            self.log_event(f"Xử lý {self.frame_count} frames, {self.total_faces_detected} faces")
        
        # Cập nhật timeline mỗi 30 frames (1s)
        if self.frame_count % 30 == 0:
            score = calculate_engagement_score(self.state_counts)
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.engagement_timeline.append({'time': elapsed, 'score': score})
            
            # Cập nhật timeline chart
            chart_html = generate_timeline_chart_html(self.engagement_timeline)
            self.timeline_chart.setHtml(chart_html)
        
        self.update_metrics_display(rois_info)
    
    def predict_engagement(self, roi):
        """
        Predict engagement state using CNN model
        Return: (state, confidence)
        """
        if self.cnn_model is None or self.cnn_model.model is None:
            return None, 0.0
        
        try:
            state, confidence = self.cnn_model.predict(roi)
            return state, confidence
        except Exception as e:
            return None, 0.0
    
    def update_metrics_display(self, rois_info=[]):
        scrollbar = self.metrics_text.verticalScrollBar()
        scroll_pos = scrollbar.value()
        
        html = generate_metrics_html(
            self.frame_count, 
            self.total_faces_detected, 
            rois_info,
            self.engaged_count,
            self.not_engaged_count,
            self.state_counts,
            self.engagement_timeline
        )
        self.metrics_text.setHtml(html)
        
        scrollbar.setValue(scroll_pos)
    
    def apply_styles(self):
        self.setStyleSheet(MAIN_WINDOW_STYLE)
    

    
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = StudentEngagementApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
