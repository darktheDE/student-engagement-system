"""
Module xử lý và làm sạch dataset Student Engagement.
Bao gồm các hàm để xử lý hàng loạt ảnh trong dataset.
"""

import os
import cv2
import numpy as np
from .preprocessing import (
    rgb_to_gray, 
    gaussian_filter, 
    histogram_equalization, 
    resize_image
)


# Đường dẫn gốc
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_ROOT = os.path.join(ROOT, "data", "raw", "Student-engagement-dataset")
DST_ROOT = os.path.join(ROOT, "data", "processed", "Student-engagement-dataset-clean")


def preprocess_for_dataset(img_bgr: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý một ảnh cho dataset.
    
    Pipeline xử lý:
    1. Chuyển sang ảnh xám
    2. Lọc Gaussian (giảm nhiễu)
    3. Cân bằng histogram (tăng độ tương phản)
    4. Resize về 48x48
    
    Args:
        img_bgr: Ảnh BGR đầu vào
        
    Returns:
        Ảnh đã xử lý (48x48, grayscale)
    """
    gray = rgb_to_gray(img_bgr)
    filtered = gaussian_filter(gray, size=5, sigma=1.0)
    equalized = histogram_equalization(filtered)
    resized = resize_image(equalized, new_size=48)
    return resized


def clean_dataset():
    """
    Xử lý và làm sạch toàn bộ dataset.
    
    Duyệt qua tất cả ảnh trong SRC_ROOT (data/raw/Student-engagement-dataset),
    áp dụng tiền xử lý và lưu vào DST_ROOT (data/processed/Student-engagement-dataset-clean).
    
    Cấu trúc thư mục:
    - Engaged/
        - confused/
        - engaged/
        - frustrated/
    - Not engaged/
        - bored/
        - drowsy/
        - Looking Away/
    """
    os.makedirs(DST_ROOT, exist_ok=True)

    for group in os.listdir(SRC_ROOT):          # Engaged, Not engaged
        group_src = os.path.join(SRC_ROOT, group)
        group_dst = os.path.join(DST_ROOT, group)
        os.makedirs(group_dst, exist_ok=True)

        for subclass in os.listdir(group_src):  # confused, bored, ...
            subclass_src = os.path.join(group_src, subclass)
            subclass_dst = os.path.join(group_dst, subclass)
            os.makedirs(subclass_dst, exist_ok=True)

            for fname in os.listdir(subclass_src):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(subclass_src, fname)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"❌ Lỗi đọc ảnh: {img_path}")
                    continue

                processed = preprocess_for_dataset(img)
                save_path = os.path.join(subclass_dst, fname)
                cv2.imwrite(save_path, processed)

    print("✅ Đã xử lý xong toàn bộ dataset!")


if __name__ == "__main__":
    clean_dataset()
