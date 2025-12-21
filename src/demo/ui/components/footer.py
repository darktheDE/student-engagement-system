"""
Footer Component - Status bar at bottom
"""
import tkinter as tk
from src.demo.config import COLOR_BACKGROUND


class FooterComponent:
    """Footer with status bar"""
    
    def __init__(self, parent):
        """
        Initialize footer component
        
        Args:
            parent: Parent tkinter widget
        """
        self.parent = parent
        self.frame = None
        self.status_lbl = None
        self._build()
    
    def _build(self):
        """Build footer UI"""
        self.frame = tk.Frame(self.parent, bg=COLOR_BACKGROUND, pady=10)
        self.frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_lbl = tk.Label(
            self.frame,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_lbl.pack(fill=tk.X)
    
    def update_status(self, text):
        """
        Update status text
        
        Args:
            text: Status message to display
        """
        if self.status_lbl:
            self.status_lbl.config(text=text)
    
    def get_frame(self):
        """Get the footer frame"""
        return self.frame
