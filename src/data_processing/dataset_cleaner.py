
import os
import sys
import cv2
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .preprocessing import (
    rgb_to_gray, 
    gaussian_filter, 
    histogram_equalization, 
    resize_image,
    center_crop_roi
)
from face_detection import FaceDetector


# Đường dẫn gốc
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_ROOT = os.path.join(ROOT, "data", "raw", "Student-engagement-dataset")
DST_ROOT = os.path.join(ROOT, "data", "processed", "Student-engagement-dataset-clean")

# Tiền xử lý một ROI (khuôn mặt đã cắt) cho dataset
def preprocess_roi(roi_bgr: np.ndarray, target_size: int = 128) -> np.ndarray:
   
    # 1. Chuyển BGR → Gray
    gray = rgb_to_gray(roi_bgr)
    
    # 2. Gaussian filter (giảm nhiễu)
    filtered = gaussian_filter(gray, size=5, sigma=1.0)
    
    # 3. Histogram equalization (tăng contrast)
    equalized = histogram_equalization(filtered)
    
    # 4. Resize về target_size x target_size
    resized = resize_image(equalized, new_size=target_size)
    
    return resized

# Tiền xử lý một ảnh cho dataset với multi-stage detection
def preprocess_for_dataset(img_bgr: np.ndarray, detector: FaceDetector, target_size: int = 128) -> tuple:
    
    # Stage 1: Thử Haar Cascade
    faces = detector.detect_faces(img_bgr)
    
    if len(faces) > 0:
        # CÓ DETECT ĐƯỢC MẶT: Cắt ROI
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        
        try:
            roi = detector.extract_roi(
                img_bgr,
                largest_face,
                padding=10,
                adaptive_padding=True
            )
            status = "haar_ok"
        except Exception:
            # Lỗi extract ROI, fallback center-crop
            roi = center_crop_roi(img_bgr, crop_ratio=0.7)
            status = "fallback_centercrop"
    else:
        # Stage 2: KHÔNG DETECT ĐƯỢC MẶT - Fallback center-crop
        roi = center_crop_roi(img_bgr, crop_ratio=0.7)
        status = "fallback_centercrop"
    
    # Stage 3: Tiền xử lý ROI: Gray → Gaussian → Histogram → Resize
    processed = preprocess_roi(roi, target_size=target_size)
    
    return processed, status

# Làm sạch toàn bộ dataset
def clean_dataset(target_size: int = 128):
     
    print("Bắt đầu xử lý dataset...")
    print(f"Source: {SRC_ROOT}")
    print(f"Destination: {DST_ROOT}")
    print(f"Target size: {target_size}x{target_size}")
    print("-" * 60)
    
    # Khởi tạo Face Detector
    print("Khởi tạo Face Detector...")
    detector = FaceDetector(use_dnn=False)
    print("Face Detector đã sẵn sàng\n")
    
    os.makedirs(DST_ROOT, exist_ok=True)
    
    total_images = 0
    processed_images = 0
    failed_images = 0
    haar_ok_count = 0
    fallback_count = 0

    for group in os.listdir(SRC_ROOT):          # Engaged, Not engaged
        group_src = os.path.join(SRC_ROOT, group)
        if not os.path.isdir(group_src):
            continue
            
        group_dst = os.path.join(DST_ROOT, group)
        os.makedirs(group_dst, exist_ok=True)
        
        print(f"\nĐang xử lý: {group}")

        for subclass in os.listdir(group_src):  # confused, bored, ...
            subclass_src = os.path.join(group_src, subclass)
            if not os.path.isdir(subclass_src):
                continue
                
            subclass_dst = os.path.join(group_dst, subclass)
            os.makedirs(subclass_dst, exist_ok=True)
            
            print(f" {subclass}: ", end="", flush=True)
            
            subclass_total = 0
            subclass_processed = 0

            for fname in os.listdir(subclass_src):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                total_images += 1
                subclass_total += 1
                img_path = os.path.join(subclass_src, fname)
                
                try:
                    # Đọc ảnh
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"\n  Lỗi đọc file: {fname}")
                        failed_images += 1
                        continue
                    
                    # Kiểm tra ảnh có hợp lệ không
                    if img.size == 0:
                        print(f"\n  Ảnh rỗng: {fname}")
                        failed_images += 1
                        continue

                    # Xử lý ảnh với Haar + Center-crop fallback
                    processed, status = preprocess_for_dataset(img, detector, target_size=target_size)
                    
                    # Đếm trạng thái
                    if status == "haar_ok":
                        haar_ok_count += 1
                    else:
                        fallback_count += 1
                    
                    if processed is None or processed.size == 0:
                        print(f"\n  Lỗi xử lý: {fname}")
                        failed_images += 1
                        continue
                    
                    # Lưu ảnh đã xử lý
                    save_path = os.path.join(subclass_dst, fname)
                    success = cv2.imwrite(save_path, processed)
                    
                    if not success:
                        print(f"\n  Lỗi lưu file: {fname}")
                        failed_images += 1
                        continue
                    
                    processed_images += 1
                    subclass_processed += 1
                    
                except Exception as e:
                    print(f"\n  Exception xử lý {fname}: {str(e)}")
                    failed_images += 1
                    continue
            
            print(f"{subclass_processed}/{subclass_total} ảnh")

    print("\n" + "=" * 60)
    print("ĐÃ XỬ LÝ XONG TOÀN BỘ DATASET!")
    print(f"Thống kê:")
    print(f"   • Tổng số ảnh: {total_images}")
    print(f"   • Xử lý thành công: {processed_images} ({processed_images*100//total_images if total_images > 0 else 0}%)")
    print(f"   • Lỗi đọc ảnh: {failed_images}")
    print(f"\nChi tiết phương pháp detect:")
    print(f"   • Haar Cascade (detect mặt): {haar_ok_count} ({haar_ok_count*100//processed_images if processed_images > 0 else 0}%)")
    print(f"   • Fallback Center-crop: {fallback_count} ({fallback_count*100//processed_images if processed_images > 0 else 0}%)")
    print("=" * 60)


if __name__ == "__main__":
    clean_dataset()
