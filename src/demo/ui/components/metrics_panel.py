"""
Metrics Panel Component - Dynamic metrics display in sidebar
"""
import tkinter as tk
from tkinter import ttk
from src.demo.config import COLOR_WHITE, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_DARK, COLOR_DIVIDER, COLOR_INFO


class MetricsPanelComponent:
    """Dynamic metrics panel that changes based on view mode"""
    
    def __init__(self, parent):
        """
        Initialize metrics panel component
        
        Args:
            parent: Parent tkinter widget (usually sidebar metrics container)
        """
        self.parent = parent
        
        # UI element references
        self.lbl_engagement = None
        self.progress_var = None
        self.progress_bar = None
        self.lbl_breakdown = None
        self.lbl_confidence = None
        
        # Comparison mode labels
        self.lbl_left_engagement = None
        self.lbl_left_breakdown = None
        self.lbl_right_engagement = None
        self.lbl_right_breakdown = None
        self.lbl_agreement = None
        
        # Stats mode labels
        self.stats_labels = {}
    
    def build_single_metrics(self):
        """Build metrics for single model view"""
        self._clear_parent()
        
        tk.Label(
            self.parent,
            text="MỨC ĐỘ HỨNG THÚ",
            font=("Arial", 12, "bold"),
            bg=COLOR_WHITE,
            fg=COLOR_PRIMARY
        ).pack(pady=8)
        
        self.lbl_engagement = tk.Label(
            self.parent,
            text="0.0%",
            font=("Arial", 42, "bold"),
            bg=COLOR_WHITE,
            fg=COLOR_INFO
        )
        self.lbl_engagement.pack(pady=8)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.parent,
            variable=self.progress_var,
            maximum=100,
            length=250,
            mode='determinate'
        )
        self.progress_bar.pack(pady=8)
        
        self.lbl_breakdown = tk.Label(
            self.parent,
            text="",
            font=("Arial", 9, "italic"),
            bg=COLOR_WHITE,
            fg=COLOR_DARK,
            wraplength=280,
            justify=tk.LEFT
        )
        self.lbl_breakdown.pack(pady=5)
        
        conf_frame = tk.Frame(self.parent, bg=COLOR_WHITE)
        conf_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(
            conf_frame,
            text="Độ tin cậy:",
            font=("Arial", 9),
            bg=COLOR_WHITE,
            fg=COLOR_DARK
        ).pack(side=tk.LEFT)
        
        self.lbl_confidence = tk.Label(
            conf_frame,
            text="N/A",
            font=("Arial", 9, "bold"),
            bg=COLOR_WHITE,
            fg=COLOR_PRIMARY
        )
        self.lbl_confidence.pack(side=tk.RIGHT)
    
    def build_comparison_metrics(self):
        """Build metrics for comparison view"""
        self._clear_parent()
        
        tk.Label(
            self.parent,
            text="SO SÁNH MODELS",
            font=("Arial", 12, "bold"),
            bg=COLOR_WHITE,
            fg=COLOR_PRIMARY
        ).pack(pady=8)
        
        # Left model
        left_frame = tk.LabelFrame(
            self.parent,
            text="Model Trái",
            font=("Arial", 9, "bold"),
            bg=COLOR_WHITE,
            fg=COLOR_INFO,
            padx=8,
            pady=5
        )
        left_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_left_engagement = self._create_inline_metric(left_frame, "Hứng thú:", "0%")
        self.lbl_left_breakdown = tk.Label(
            left_frame,
            text="",
            font=("Arial", 8),
            bg=COLOR_WHITE,
            fg=COLOR_DARK,
            wraplength=250,
            justify=tk.LEFT
        )
        self.lbl_left_breakdown.pack(fill=tk.X, pady=2)
        
        # Right model
        right_frame = tk.LabelFrame(
            self.parent,
            text="Model Phải",
            font=("Arial", 9, "bold"),
            bg=COLOR_WHITE,
            fg="#e67e22",
            padx=8,
            pady=5
        )
        right_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_right_engagement = self._create_inline_metric(right_frame, "Hứng thú:", "0%")
        self.lbl_right_breakdown = tk.Label(
            right_frame,
            text="",
            font=("Arial", 8),
            bg=COLOR_WHITE,
            fg=COLOR_DARK,
            wraplength=250,
            justify=tk.LEFT
        )
        self.lbl_right_breakdown.pack(fill=tk.X, pady=2)
        
        # Agreement
        tk.Frame(self.parent, height=2, bg=COLOR_DIVIDER).pack(fill=tk.X, pady=8)
        self.lbl_agreement = self._create_inline_metric(self.parent, "Đồng thuận:", "0%")
    
    def build_stats_metrics(self):
        """Build metrics for stats view"""
        self._clear_parent()
        
        tk.Label(
            self.parent,
            text="THỐNG KÊ CHI TIẾT",
            font=("Arial", 12, "bold"),
            bg=COLOR_WHITE,
            fg=COLOR_PRIMARY
        ).pack(pady=8)
        
        self.stats_labels = {}
        
        for model_id, model_name in [("cnn_feature", "CNN+SVM"), 
                                     ("cnn_softmax", "CNN Thuần"), 
                                     ("hog", "HOG+SVM")]:
            frame = tk.LabelFrame(
                self.parent,
                text=model_name,
                font=("Arial", 9, "bold"),
                bg=COLOR_WHITE,
                fg=COLOR_SECONDARY,
                padx=8,
                pady=5
            )
            frame.pack(fill=tk.X, pady=3)
            
            self.stats_labels[model_id] = {
                'engagement': self._create_inline_metric(frame, "Hứng thú:", "0%"),
                'time': self._create_inline_metric(frame, "Thời gian:", "0ms")
            }
    
    def _create_inline_metric(self, parent, label_text, value_text):
        """Create an inline metric display"""
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
    
    def _clear_parent(self):
        """Clear all widgets from parent"""
        for widget in self.parent.winfo_children():
            widget.destroy()
