# Student Engagement Demo - Refactored Version

## 📁 Cấu trúc Module

### Core Logic (`core/`)
Chứa business logic chính của ứng dụng:
- **model_manager.py**: Quản lý loading và storage của ML models
- **predictor.py**: Xử lý prediction logic cho tất cả models
- **video_processor.py**: Xử lý camera, video frames, và face detection

### UI Components (`ui/`)
Chứa tất cả UI components:
- **app.py**: Main application class - điều phối toàn bộ ứng dụng
- **components/**: Các UI components độc lập
  - `header.py`: Header với tiêu đề và thông tin nhóm
  - `sidebar.py`: Sidebar với metrics và controls
  - `video_panel.py`: Video display với mode controls
  - `metrics_panel.py`: Dynamic metrics display
  - `footer.py`: Status bar

### Utilities (`utils_new/`)
Các helper functions và utilities:
- **preprocessing.py**: Image preprocessing cho models
- **visualization.py**: Drawing predictions lên frames
- **metrics.py**: Calculations cho engagement rate, confidence, etc.
- **mapping.py**: Label và binary mapping
- **image_utils.py**: Image conversion utilities

### Configuration (`config.py`)
Centralized configuration file chứa:
- Model paths
- Display settings
- Label mappings
- UI colors
- Detection thresholds

## 🚀 Cách sử dụng

### Chạy ứng dụng:

```bash
# Từ project root
python src/demo/main.py

# Hoặc với PYTHONPATH (Windows)
$env:PYTHONPATH="<path_to_project_root>"
python src/demo/main.py
```

### Import các modules:

```python
# Import core components
from src.demo.core import ModelManager, Predictor, VideoProcessor

# Import UI
from src.demo.ui import EngagementApp

# Import utilities
from src.demo.utils_new import (
    preprocess_for_cnn,
    draw_prediction_on_frame,
    calculate_engagement_rate
)

# Import config
from src.demo.config import LABEL_MAP, BINARY_MAP
```

## 🎨 Design Patterns

### 1. **Component-Based Architecture**
Mỗi UI component là một class độc lập, tự quản lý state và rendering.

### 2. **Dependency Injection**
- `Predictor` nhận `ModelManager` qua constructor
- `VideoProcessor` nhận `FaceDetector` qua constructor
- Dễ testing và mocking

### 3. **Single Responsibility Principle**
Mỗi class có một trách nhiệm duy nhất:
- `ModelManager`: Loading models
- `Predictor`: Making predictions
- `VideoProcessor`: Processing video
- `EngagementApp`: Orchestrating application

### 4. **Observer Pattern**
UI components register callbacks để nhận updates khi có thay đổi.

## 🔧 Mở rộng

### Thêm model mới:

1. Update `config.py` với model path
2. Thêm loading method trong `ModelManager`
3. Thêm prediction method trong `Predictor`
4. Update UI selectors nếu cần

### Thêm view mode mới:

1. Thêm option trong `VideoPanelComponent`
2. Thêm layout method trong `MetricsPanelComponent`
3. Thêm visualization logic trong `EngagementApp`

### Thêm metric mới:

1. Thêm calculation function vào `utils_new/metrics.py`
2. Update UI component để hiển thị metric
3. Update update loop để calculate metric

## 📝 Notes

- Tất cả imports sử dụng absolute paths (`from src.demo...`)
- Config centralized trong `config.py`
- UI components hoàn toàn độc lập với business logic
- Video processing tách biệt khỏi prediction logic

## 🐛 Debugging

Nếu gặp `ModuleNotFoundError`:
```bash
# Set PYTHONPATH trước khi chạy
$env:PYTHONPATH="<project_root>"
python src/demo/main.py
```

Nếu camera không hoạt động:
- Kiểm tra quyền truy cập camera
- Thử chạy với admin privileges
- Check `VideoProcessor.initialize_camera()` logs

## 📚 Documentation

Xem thêm:
- [REFACTORING_COMPLETE.md](../../docs/REFACTORING_COMPLETE.md) - Chi tiết về quá trình refactoring
- [UI_GUIDE.md](../../docs/UI_GUIDE.md) - Hướng dẫn UI
- [flow.md](../../docs/flow.md) - Application flow

---

**Version**: 2.0.0 (Refactored)  
**Group**: 16  
**Date**: December 2025
