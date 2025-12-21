"""
Image utility functions for conversions and display
"""
import cv2
import PIL.Image
import PIL.ImageTk


def convert_cv2_to_tkinter(opencv_img, width, height):
    """
    Convert OpenCV image to Tkinter PhotoImage
    
    Args:
        opencv_img: OpenCV image (numpy array, BGR)
        width: Target width
        height: Target height
        
    Returns:
        PIL.ImageTk.PhotoImage: Image ready for Tkinter display
    """
    # Resize
    img_resized = cv2.resize(opencv_img, (width, height))
    
    # Convert BGR to RGB
    color = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL and then to PhotoImage
    image = PIL.Image.fromarray(color)
    photo = PIL.ImageTk.PhotoImage(image)
    
    return photo
