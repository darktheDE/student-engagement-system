import os
import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Hàm chuyển ảnh BGR sang ảnh xám

def rgb_to_gray(img_bgr: np.ndarray) -> np.ndarray:
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

# Hàm lọc trung bình trên ảnh xám

def mean_filter(gray: np.ndarray, ksize: int = 7) -> np.ndarray:
   
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    filtered = convolve2d(gray, kernel, mode='same', boundary='symm')
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered

def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
   
    s = (size - 1) / 2
    ax = np.linspace(-s, s, size)
    g1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    kernel = np.outer(g1d, g1d)
    kernel = kernel / kernel.sum()
    return kernel

# Hàm lọc Gaussian trên ảnh xám

def gaussian_filter(gray: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    
    k = gaussian_kernel(size, sigma)
    filtered = convolve2d(gray, k, mode='same', boundary='symm')
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered

# Hàm lọc trung vị trên ảnh xám

def median_filter(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
   
    return cv2.medianBlur(gray, ksize)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

SRC_ROOT = os.path.join(ROOT, "data", "raw", "Student-engagement-dataset")
DST_ROOT = os.path.join(ROOT, "data", "processed", "Student-engagement-dataset-clean")

def preprocess_for_dataset(img_bgr: np.ndarray) -> np.ndarray:
  
    gray = rgb_to_gray(img_bgr)
    filtered = gaussian_filter(gray, size=5, sigma=1.0)
    return filtered

def clean_dataset():
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
                    print("Lỗi đọc ảnh:", img_path)
                    continue

                processed = preprocess_for_dataset(img)
                save_path = os.path.join(subclass_dst, fname)
                cv2.imwrite(save_path, processed)

    print("✅ Đã xử lý xong toàn bộ dataset!")


if __name__ == "__main__":
    # demo_filters(r"...đường_dẫn_1_ảnh...")   # chạy demo cho báo cáo
    clean_dataset()                             # xử lý full dataset
