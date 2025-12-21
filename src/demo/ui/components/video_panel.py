"""
Video Panel Component - Main video display area with controls
"""
import tkinter as tk
from src.demo.config import COLOR_WHITE, COLOR_SECONDARY, COLOR_PRIMARY, COLOR_BLACK


class VideoPanelComponent:
    """Video display panel with mode controls"""
    
    def __init__(self, parent):
        """
        Initialize video panel component
        
        Args:
            parent: Parent tkinter widget
        """
        self.parent = parent
        self.frame = None
        
        # Variables
        self.view_mode = tk.StringVar(value="single")
        self.show_preprocessed = tk.BooleanVar(value=False)
        self.model_left = tk.StringVar(value="cnn_feature")
        self.model_right = tk.StringVar(value="hog")
        
        # UI elements
        self.model_selector_frame = None
        self.video_display_frame = None
        self.canvas_single = None
        self.canvas_left = None
        self.canvas_right = None
        
        # Callbacks
        self.on_mode_change_callback = None
        
        self._build()
    
    def _build(self):
        """Build video panel UI"""
        self.frame = tk.Frame(self.parent, bg=COLOR_WHITE, bd=2, relief=tk.GROOVE)
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Control Bar
        control_frame = tk.Frame(self.frame, bg=COLOR_SECONDARY, pady=8)
        control_frame.pack(fill=tk.X)
        
        # Mode selector
        tk.Label(
            control_frame,
            text="Chế độ:",
            font=("Arial", 10, "bold"),
            bg=COLOR_SECONDARY,
            fg="white"
        ).pack(side=tk.LEFT, padx=10)
        
        modes = [("Đơn", "single"), ("So sánh", "comparison"), ("Thống kê", "stats")]
        for text, value in modes:
            rb = tk.Radiobutton(
                control_frame,
                text=text,
                variable=self.view_mode,
                value=value,
                font=("Arial", 9),
                bg=COLOR_SECONDARY,
                fg="white",
                selectcolor=COLOR_PRIMARY,
                activebackground=COLOR_SECONDARY,
                activeforeground="white",
                command=self._on_mode_change
            )
            rb.pack(side=tk.LEFT, padx=3)
        
        # Preprocessing toggle
        tk.Frame(control_frame, width=20, bg=COLOR_SECONDARY).pack(side=tk.LEFT)
        tk.Checkbutton(
            control_frame,
            text="📸 Hiển thị ảnh đã xử lý",
            variable=self.show_preprocessed,
            font=("Arial", 9),
            bg=COLOR_SECONDARY,
            fg="white",
            selectcolor=COLOR_PRIMARY,
            activebackground=COLOR_SECONDARY,
            activeforeground="white"
        ).pack(side=tk.LEFT, padx=10)
        
        # Model Selectors (dynamic)
        self.model_selector_frame = tk.Frame(control_frame, bg=COLOR_SECONDARY)
        self.model_selector_frame.pack(side=tk.LEFT, padx=20)
        
        # Video Display Container
        self.video_display_frame = tk.Frame(self.frame, bg=COLOR_BLACK)
        self.video_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvases
        self.canvas_single = tk.Label(self.video_display_frame, bg=COLOR_BLACK)
        self.canvas_left = tk.Label(self.video_display_frame, bg=COLOR_BLACK)
        self.canvas_right = tk.Label(self.video_display_frame, bg=COLOR_BLACK)
        
        # Initial layout
        self._update_canvas_layout()
        self._update_model_selectors()
    
    def _on_mode_change(self):
        """Handle mode change"""
        self._update_canvas_layout()
        self._update_model_selectors()
        
        if self.on_mode_change_callback:
            self.on_mode_change_callback()
    
    def _update_canvas_layout(self):
        """Update canvas layout based on view mode"""
        # Hide all canvases
        self.canvas_single.pack_forget()
        self.canvas_left.pack_forget()
        self.canvas_right.pack_forget()
        
        mode = self.view_mode.get()
        
        if mode == "single":
            self.canvas_single.pack(fill=tk.BOTH, expand=True)
        elif mode == "comparison":
            self.canvas_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
            self.canvas_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
        elif mode == "stats":
            self.canvas_single.pack(fill=tk.BOTH, expand=True)
    
    def _update_model_selectors(self):
        """Update model selector UI based on mode"""
        # Clear existing selectors
        for widget in self.model_selector_frame.winfo_children():
            widget.destroy()
        
        mode = self.view_mode.get()
        
        if mode == "single":
            self._build_single_selector()
        elif mode == "comparison":
            self._build_comparison_selectors()
    
    def _build_single_selector(self):
        """Build single model selector"""
        tk.Label(
            self.model_selector_frame,
            text="Model:",
            font=("Arial", 9, "bold"),
            bg=COLOR_SECONDARY,
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        models = [("CNN+SVM", "cnn_feature"), ("CNN Thuần", "cnn_softmax"), ("HOG+SVM", "hog")]
        for text, value in models:
            rb = tk.Radiobutton(
                self.model_selector_frame,
                text=text,
                variable=self.model_left,
                value=value,
                font=("Arial", 8),
                bg=COLOR_SECONDARY,
                fg="white",
                selectcolor=COLOR_PRIMARY,
                activebackground=COLOR_SECONDARY,
                activeforeground="white"
            )
            rb.pack(side=tk.LEFT, padx=2)
    
    def _build_comparison_selectors(self):
        """Build comparison mode selectors"""
        tk.Label(
            self.model_selector_frame,
            text="Trái:",
            font=("Arial", 9, "bold"),
            bg=COLOR_SECONDARY,
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        models = [("CNN+SVM", "cnn_feature"), ("CNN", "cnn_softmax"), ("HOG", "hog")]
        for text, value in models:
            rb = tk.Radiobutton(
                self.model_selector_frame,
                text=text,
                variable=self.model_left,
                value=value,
                font=("Arial", 8),
                bg=COLOR_SECONDARY,
                fg="white",
                selectcolor=COLOR_PRIMARY,
                activebackground=COLOR_SECONDARY,
                activeforeground="white"
            )
            rb.pack(side=tk.LEFT, padx=2)
        
        tk.Label(
            self.model_selector_frame,
            text=" | Phải:",
            font=("Arial", 9, "bold"),
            bg=COLOR_SECONDARY,
            fg="white"
        ).pack(side=tk.LEFT, padx=5)
        
        for text, value in models:
            rb = tk.Radiobutton(
                self.model_selector_frame,
                text=text,
                variable=self.model_right,
                value=value,
                font=("Arial", 8),
                bg=COLOR_SECONDARY,
                fg="white",
                selectcolor=COLOR_PRIMARY,
                activebackground=COLOR_SECONDARY,
                activeforeground="white"
            )
            rb.pack(side=tk.LEFT, padx=2)
    
    def set_mode_change_callback(self, callback):
        """Set callback for mode change"""
        self.on_mode_change_callback = callback
    
    def get_frame(self):
        """Get the video panel frame"""
        return self.frame
    
    def get_canvases(self):
        """Get all canvas widgets"""
        return {
            'single': self.canvas_single,
            'left': self.canvas_left,
            'right': self.canvas_right
        }
