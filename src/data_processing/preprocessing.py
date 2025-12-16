

import cv2
import numpy as np
from scipy.signal import convolve2d

# Chuyển ảnh BGR sang ảnh xám
def rgb_to_gray(img_bgr: np.ndarray) -> np.ndarray:
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

# Tạo kernel Gaussian
def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
  
    s = (size - 1) / 2
    ax = np.linspace(-s, s, size)
    g1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    kernel = np.outer(g1d, g1d)
    kernel = kernel / kernel.sum()
    return kernel

# Lọc Gaussian trên ảnh xám
def gaussian_filter(gray: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
   
    k = gaussian_kernel(size, sigma)
    filtered = convolve2d(gray, k, mode='same', boundary='symm')
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    return filtered

# Cân bằng histogram trên ảnh xám
def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    
    h, w = gray.shape
    total_pixels = h * w

    # 1. Tính histogram
    hist = np.zeros(256, dtype=np.int32)
    for y in range(h):
        for x in range(w):
            hist[gray[y, x]] += 1

    # 2. Tính CDF
    cdf = np.zeros(256, dtype=np.int32)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # 3. Chuẩn hóa CDF
    cdf_min = next((v for v in cdf if v > 0), 0)
    lut = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        lut[i] = np.clip(
            round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255),
            0, 255
        )

    # 4. Ánh xạ ảnh
    equalized = np.zeros_like(gray, dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            equalized[y, x] = lut[gray[y, x]]

    return equalized

# Cắt vùng trung tâm của ảnh (fallback khi không detect được mặt)
def center_crop_roi(img_bgr: np.ndarray, crop_ratio: float = 0.7) -> np.ndarray:
   
    h, w = img_bgr.shape[:2]
    
    # Tính toạ độ crop (bias về trên)
    top = int(h * 0.05)  # 5% từ trên
    bottom = int(h * (0.05 + crop_ratio))  # 75% từ trên (5% + 70%)
    left = int(w * (1 - crop_ratio) / 2)  # Center theo chiều ngang
    right = int(w * (1 - (1 - crop_ratio) / 2))
    
    # Crop
    roi = img_bgr[top:bottom, left:right]
    
    return roi

# Thay đổi kích thước ảnh xám.
def resize_image(gray: np.ndarray, new_size: int = 128) -> np.ndarray:
   
    h, w = gray.shape
    resized = np.zeros((new_size, new_size), dtype=np.uint8)

    scale_x = w / new_size
    scale_y = h / new_size

    for y in range(new_size):
        for x in range(new_size):
            src_x = int(x * scale_x)
            src_y = int(y * scale_y)

            if src_x >= w:
                src_x = w - 1
            if src_y >= h:
                src_y = h - 1

            resized[y, x] = gray[src_y, src_x]

    return resized
