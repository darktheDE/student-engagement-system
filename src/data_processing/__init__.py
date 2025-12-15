"""
Data Processing Module

Module xử lý dữ liệu cho Student Engagement System.
Bao gồm các chức năng tiền xử lý ảnh và làm sạch dataset.
"""

# Import các hàm xử lý ảnh cơ bản từ preprocessing
from .preprocessing import (
    rgb_to_gray,
    gaussian_kernel,
    gaussian_filter,
    histogram_equalization,
    resize_image
)

# Import các hàm xử lý dataset từ dataset_cleaner
from .dataset_cleaner import (
    preprocess_for_dataset,
    clean_dataset,
    ROOT,
    SRC_ROOT,
    DST_ROOT
)

# Định nghĩa các public API của module
__all__ = [
    # Preprocessing functions
    'rgb_to_gray',
    'gaussian_kernel',
    'gaussian_filter',
    'histogram_equalization',
    'resize_image',
    # Dataset processing functions
    'preprocess_for_dataset',
    'clean_dataset',
    # Path constants
    'ROOT',
    'SRC_ROOT',
    'DST_ROOT',
]                           
