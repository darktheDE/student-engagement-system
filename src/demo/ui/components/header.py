"""
Header Component - Top banner with title and group info
"""
import tkinter as tk
from src.demo.config import COLOR_PRIMARY, COLOR_WARNING, COLOR_LIGHT


class HeaderComponent:
    """Header section with title and group information"""
    
    def __init__(self, parent):
        """
        Initialize header component
        
        Args:
            parent: Parent tkinter widget
        """
        self.parent = parent
        self.frame = None
        self._build()
    
    def _build(self):
        """Build header UI"""
        self.frame = tk.Frame(self.parent, bg=COLOR_PRIMARY, pady=20)
        self.frame.pack(fill=tk.X)
        
        # Title
        title = tk.Label(
            self.frame, 
            text="PHÂN LOẠI MỨC ĐỘ HỨNG THÚ HỌC TẬP CỦA SINH VIÊN\nBẰNG PHÂN TÍCH KHUÔN MẶT",
            font=("Arial", 20, "bold"),
            fg="white",
            bg=COLOR_PRIMARY,
            justify=tk.CENTER
        )
        title.pack()
        
        # Group label
        group = tk.Label(
            self.frame,
            text="Nhóm 16",
            font=("Arial", 16, "bold"),
            fg=COLOR_WARNING,
            bg=COLOR_PRIMARY
        )
        group.pack(pady=5)
        
        # Members
        members = "Huỳnh Ngọc Thạch - 23133072  |  Huỳnh Hữu Huy - 23133027  |  Đỗ Kiến Hưng - 23133030  |  Nguyễn Tấn Thành - 23133068"
        mem_lbl = tk.Label(
            self.frame,
            text=members,
            font=("Arial", 12),
            fg=COLOR_LIGHT,
            bg=COLOR_PRIMARY
        )
        mem_lbl.pack()
    
    def get_frame(self):
        """Get the header frame"""
        return self.frame
