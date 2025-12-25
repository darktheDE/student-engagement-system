
import os
import sys
import time
import cv2
import numpy as np
from scipy import signal

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.data_processing.preprocessing import (
    rgb_to_gray as custom_rgb_to_gray,
    gaussian_filter as custom_gaussian_filter,
    histogram_equalization as custom_histogram_equalization,
    resize_image as custom_resize_image
)

def opencv_pipeline(img_bgr, target_size=128):
    # 1. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur (match custom kernel size 5, sigma 1.0)
    # Custom uses a kernel of size 5 with sigma 1.0.
    # cv2.GaussianBlur(src, ksize, sigmaX)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # 3. Histogram Equalization
    equalized = cv2.equalizeHist(blurred)
    
    # 4. Resize
    # Custom resize implementation seems to hold nearest neighbor-ish logic or bilinear?
    # Let's check custom resize again... simple scaling.
    # Custom uses manual nearest neighbor scaling (int casting)
    resized = cv2.resize(equalized, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    return resized

def run_comparison(image_path):
    print(f"Reading image: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Could not read image.")
        return

    # --- Run Custom Pipeline ---
    start_custom = time.time()
    
    # Custom pipeline steps manual calls to match what dataset_cleaner does
    c_gray = custom_rgb_to_gray(img_bgr)
    c_blurred = custom_gaussian_filter(c_gray, size=5, sigma=1.0)
    c_eq = custom_histogram_equalization(c_blurred)
    c_resized = custom_resize_image(c_eq, new_size=128)
    
    end_custom = time.time()
    time_custom = (end_custom - start_custom) * 1000
    
    # --- Run OpenCV Pipeline ---
    start_cv = time.time()
    cv_result = opencv_pipeline(img_bgr, target_size=128)
    end_cv = time.time()
    time_cv = (end_cv - start_cv) * 1000
    
    # --- Comparison ---
    # MSE
    mse = np.mean((c_resized.astype("float") - cv_result.astype("float")) ** 2)
    
    print("-" * 30)
    print(f"Custom Pipeline Time: {time_custom:.2f} ms")
    print(f"OpenCV Pipeline Time: {time_cv:.2f} ms")
    print(f"Speedup Factor: {time_custom / time_cv:.2f}x")
    print(f"Mean Squared Error: {mse:.4f}")
    print("-" * 30)
    
    if mse < 10: # Threshold for "close enough" (pixel values 0-255)
        print("SUCCESS: Pipelines are compatible.")
    else:
        print("WARNING: Pipelines differ significantly.")

    # Optional: Save for visual check
    cv2.imwrite("test_custom.jpg", c_resized)
    cv2.imwrite("test_opencv.jpg", cv_result)
    print("Saved outputs to test_custom.jpg and test_opencv.jpg")

if __name__ == "__main__":
    # Use a sample image found in previous step
    # Adjust path as needed based on where the script is run
    sample_img = r"d:\HCMUTE\HCMUTE_HK5\DIPR\final\student-engagement-system\data\raw\Student-engagement-dataset\Not engaged\bored\0034.jpg"
    
    if os.path.exists(sample_img):
        run_comparison(sample_img)
    else:
        print("Sample image not found, please provide a valid path.")
