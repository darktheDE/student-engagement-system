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
    """
    Vẽ bounding boxes lên frame
    predictions: List các dict {'state': str, 'confidence': float}
    """
    rois_info = []
    
    for i, (x, y, w, h) in enumerate(faces):
        try:
            roi = detector.extract_roi(frame, (x, y, w, h), target_size=(256, 256))
            quality_score, metrics = detector.assess_face_quality(roi)
            
            # Lấy thông tin dự đoán nếu có
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
            
            # Xác định màu sắc và label
            if state:
                state_info = get_state_info(state)
                color = state_info['color_bgr']
                label_text = f"Face #{i + 1}: {state_info['display_name']} ({confidence:.0%})"
            else:
                # Fallback: Màu theo chất lượng
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
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Vẽ background cho text
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), color, -1)
            
            # Vẽ text
            cv2.putText(frame, label_text, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        except Exception as e:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
    
    return frame, rois_info


def generate_metrics_html(frame_count, total_faces_detected, rois_info, 
                         engaged_count=0, not_engaged_count=0, state_counts=None, timeline=None):
    """
    Tạo HTML hiển thị thống kê chi tiết
    state_counts: dict {'engaged': x, 'confused': y, ...}
    timeline: list [{'time': seconds, 'score': 0-10}, ...]
    """
    if frame_count == 0:
        return """
            <div style='padding: 12px; font-family: Arial;'>
                <p style='color: #95a5a6; text-align: center;'>Chưa có dữ liệu</p>
            </div>
        """
    
    avg_faces = total_faces_detected / frame_count if frame_count > 0 else 0
    
    # Tính engagement score (1-10)
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
        
        <!-- Thông tin cơ bản -->
        <div style='margin-bottom: 10px; background-color: #1a1f2e; border-radius: 6px; padding: 10px;'>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr>
                    <td style='padding: 5px; color: #95a5a6; font-size: 11px;'>Tổng frames:</td>
                    <td style='padding: 5px; color: #e5e7eb; text-align: right; font-size: 11px; font-weight: bold;'>{frame_count}</td>
                </tr>
                <tr>
                    <td style='padding: 5px; color: #95a5a6; font-size: 11px;'>Tổng faces:</td>
                    <td style='padding: 5px; color: #e5e7eb; text-align: right; font-size: 11px; font-weight: bold;'>{total_faces_detected}</td>
                </tr>
                <tr>
                    <td style='padding: 5px; color: #95a5a6; font-size: 11px;'>TB faces/frame:</td>
                    <td style='padding: 5px; color: #e5e7eb; text-align: right; font-size: 11px; font-weight: bold;'>{avg_faces:.2f}</td>
                </tr>
            </table>
        </div>
        
        <!-- Chi tiết trạng thái -->
        <div style='margin-bottom: 10px; background-color: #1a1f2e; border-radius: 6px; padding: 10px;'>
            <div style='color: #60a5fa; font-weight: bold; margin-bottom: 8px; font-size: 11px;'>PHÂN BỐ TRẠNG THÁI</div>
    """
    
    # Hiển thị tất cả 6 trạng thái
    from engagement_config import ENGAGEMENT_STATES
    if state_counts:
        for state_key, state_info in ENGAGEMENT_STATES.items():
            count = state_counts.get(state_key, 0)
            percentage = (count / total_faces_detected * 100) if total_faces_detected > 0 else 0
            html += f"""
            <div style='margin-bottom: 6px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 3px;'>
                    <span style='color: {state_info['color_hex']}; font-size: 10px; font-weight: 600;'>{state_info['display_name']}</span>
                    <span style='color: #95a5a6; font-size: 10px;'>{count} ({percentage:.1f}%)</span>
                </div>
                <div style='width: 100%; height: 6px; background-color: #0f1419; border-radius: 3px; overflow: hidden;'>
                    <div style='width: {percentage}%; height: 100%; background-color: {state_info['color_hex']}; transition: width 0.3s;'></div>
                </div>
            </div>
            """
    
    html += """
        </div>
    </div>
    """
    
    return html
