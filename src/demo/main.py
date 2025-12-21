"""
Main Entry Point - Student Engagement Classification Demo
Refactored version with modular architecture

Run this file to start the application:
    python src/demo/main.py
"""
import sys
import os

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
demo_dir = current_dir
src_dir = os.path.dirname(demo_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tkinter as tk
from src.demo.ui import EngagementApp


def main():
    """Main entry point"""
    root = tk.Tk()
    app = EngagementApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
