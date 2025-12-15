from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


def create_header():
    header_frame = QFrame()
    header_frame.setObjectName("headerFrame")
    header_frame.setMaximumHeight(95)
    header_layout = QVBoxLayout(header_frame)
    header_layout.setContentsMargins(12, 8, 12, 8)
    header_layout.setSpacing(2)
    
    title = QLabel("HỆ THỐNG PHÂN LOẠI MỨC ĐỘ HỨNG THÚ HỌC TẬP")
    title.setObjectName("titleLabel")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    header_layout.addWidget(title)
    
    team_info = QLabel("Nhóm 16 | Thành viên: Đỗ Kiến Hưng - Huỳnh Ngọc Thạch - Huỳnh Hữu Huy - Nguyễn Tấn Thành")
    team_info.setObjectName("teamInfoLabel")
    team_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
    header_layout.addWidget(team_info)
    
    return header_frame
