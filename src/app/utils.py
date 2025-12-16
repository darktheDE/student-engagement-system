import cv2
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

try:
    from .engagement_config import get_state_info, calculate_engagement_score
except ImportError:
    from engagement_config import get_state_info, calculate_engagement_score


def cv_to_pixmap(frame, label_size):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_image)
    return pixmap.scaled(
        label_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )


def draw_face_boxes(frame, faces, detector, predictions=None):

    rois_info = []
    
    for i, (x, y, w, h) in enumerate(faces):
        try:
            roi = detector.extract_roi(frame, (x, y, w, h), target_size=(256, 256))
            quality_score, metrics = detector.assess_face_quality(roi)
            
            state = None
            confidence = 0.0
            if predictions and i < len(predictions):
                state = predictions[i].get('state')
                confidence = predictions[i].get('confidence', 0.0)
            
            rois_info.append({
                'id': i + 1,
                'bbox': (x, y, w, h),
                'size': (w, h),
                'quality': quality_score,
                'state': state,
                'confidence': confidence
            })
            
            if state:
                state_info = get_state_info(state)
                color = state_info['color_bgr']
                label_text = f"Face #{i + 1}: {state_info['display_name']} ({confidence:.0%})"
            else:
                if quality_score >= 0.7:
                    color = (0, 200, 0)
                    status = "Tốt"
                elif quality_score >= 0.5:
                    color = (0, 165, 255)
                    status = "TB"
                else:
                    color = (0, 0, 255)
                    status = "Kém"
                label_text = f"Face #{i + 1} [{status}]"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            
            cv2.putText(frame, label_text, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        except Exception as e:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
    
    return frame, rois_info


def generate_metrics_html(frame_count, total_faces_detected, rois_info, 
                         engaged_count=0, not_engaged_count=0, state_counts=None, timeline=None):

    if frame_count == 0:
        return """
            <div style='padding: 12px; font-family: Arial;'>
                <p style='color: #95a5a6; text-align: center;'>Chưa có dữ liệu</p>
            </div>
        """
    
    avg_faces = total_faces_detected / frame_count if frame_count > 0 else 0
    
    engagement_score = 0
    if state_counts:
        engagement_score = calculate_engagement_score(state_counts)
    
    html = f"""
    <div style='padding: 8px; font-family: Arial; line-height: 1.5; font-size: 11px;'>
        
        <!-- Chỉ số hứng thú trung bình (1-10) -->
        <div style='margin-bottom: 12px; padding: 15px; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 10px; text-align: center;'>
            <div style='color: #e0e7ff; font-size: 11px; margin-bottom: 5px; font-weight: 500;'>CHỈ SỐ HỨNG THÚ TRUNG BÌNH</div>
            <div style='color: #ffffff; font-size: 40px; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{engagement_score:.1f}<span style='font-size: 20px;'>/10</span></div>
        </div>
        
        <!-- Thông tin cơ bản (removed: totals and averages) -->
        
        <!-- Chi tiết trạng thái (removed detailed headings) -->
    """

    html += f"""
        <div style='margin-bottom: 10px; background-color: #1a1f2e; border-radius: 6px; padding: 10px;'>
            <div style='color: #e5e7eb; font-size: 12px;'>Engaged: {engaged_count} &nbsp;&nbsp; Not Engaged: {not_engaged_count}</div>
        </div>
    </div>
    """
    
    return html
