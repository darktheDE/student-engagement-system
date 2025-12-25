"""
Sidebar Component - Right sidebar with stats and controls
"""
import tkinter as tk
from tkinter import ttk
from src.demo.config import (COLOR_WHITE, COLOR_PRIMARY, COLOR_SECONDARY,
                       COLOR_DARK, COLOR_DIVIDER)


class SidebarComponent:
    """Sidebar with performance metrics and controls"""
    
    def __init__(self, parent):
        """
        Initialize sidebar component
        
        Args:
            parent: Parent tkinter widget
        """
        self.parent = parent
        self.frame = None
        
        # Label references for updates
        self.lbl_fps = None
        self.lbl_process_time = None
        self.lbl_faces = None
        self.lbl_light_quality = None
        self.brightness_adjust = None
        self.metrics_container = None
        
        self._build()
    
    def _build(self):
        """Build sidebar UI with scrollbar"""
        # Main frame
        self.frame = tk.Frame(self.parent, bg=COLOR_WHITE, width=320, relief=tk.RIDGE, bd=2)
        self.frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.frame.pack_propagate(False)
        
        # Create Canvas and Scrollbar for scrollable content
        canvas = tk.Canvas(self.frame, bg=COLOR_WHITE, highlightthickness=0)
        scrollbar = tk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLOR_WHITE)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Title
        tk.Label(
            scrollable_frame,
            text="THỐNG KÊ",
            font=("Arial", 14, "bold"),
            bg=COLOR_WHITE,
            fg=COLOR_PRIMARY
        ).pack(pady=12)
        
        # Performance Section
        perf_frame = tk.LabelFrame(
            scrollable_frame,
            text="Hiệu suất",
            font=("Arial", 10, "bold"),
            bg=COLOR_WHITE,
            fg=COLOR_SECONDARY,
            padx=10,
            pady=8
        )
        perf_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.lbl_fps = self._create_inline_metric(perf_frame, "FPS:", "0")
        self.lbl_process_time = self._create_inline_metric(perf_frame, "Xử lý:", "0 ms")
        
        # Detection Section
        detect_frame = tk.LabelFrame(
            scrollable_frame,
            text="Phát hiện",
            font=("Arial", 10, "bold"),
            bg=COLOR_WHITE,
            fg=COLOR_SECONDARY,
            padx=10,
            pady=8
        )
        detect_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.lbl_faces = self._create_inline_metric(detect_frame, "Khuôn mặt:", "0")
        self.lbl_light_quality = self._create_inline_metric(detect_frame, "Ánh sáng:", "N/A")
        
        # Divider
        tk.Frame(scrollable_frame, height=2, bg=COLOR_DIVIDER).pack(fill=tk.X, padx=20, pady=12)
        
        # Metrics Container (dynamic content)
        self.metrics_container = tk.Frame(scrollable_frame, bg=COLOR_WHITE)
        self.metrics_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 20))
    
    def _create_inline_metric(self, parent, label_text, value_text):
        """
        Create an inline metric display (label: value)
        
        Args:
            parent: Parent widget
            label_text: Label text
            value_text: Initial value text
            
        Returns:
            tk.Label: Value label widget
        """
        frame = tk.Frame(parent, bg=COLOR_WHITE)
        frame.pack(fill=tk.X, pady=2)
        
        tk.Label(
            frame,
            text=label_text,
            font=("Arial", 9),
            bg=COLOR_WHITE,
            fg=COLOR_DARK
        ).pack(side=tk.LEFT)
        
        val_lbl = tk.Label(
            frame,
            text=value_text,
            font=("Arial", 9, "bold"),
            bg=COLOR_WHITE,
            fg=COLOR_PRIMARY
        )
        val_lbl.pack(side=tk.RIGHT)
        
        return val_lbl
    
    def get_frame(self):
        """Get the sidebar frame"""
        return self.frame
    
    def get_metrics_container(self):
        """Get the metrics container frame"""
        return self.metrics_container
