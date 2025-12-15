import cv2
import numpy as np
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .face_detector import FaceDetector
from . import config


class WebcamDemo:
    def __init__(self, camera_index=0, use_dnn=False):
        self.detector = FaceDetector(use_dnn=use_dnn)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        self.frame_count, self.fps = 0, 0
        self.start_time = time.time()
        self.screenshot_dir = os.path.join(os.path.dirname(__file__), 'screenshots')
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def calculate_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        return self.fps

    def process_frame(self, frame):
        faces = self.detector.detect_faces(frame)
        annotated = self.detector.draw_faces(frame, faces)
        fps = self.calculate_fps()
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                   config.FONT_FACE, config.FONT_SCALE,
                   config.COLOR_TEXT, config.FONT_THICKNESS)
        cv2.putText(annotated, f"Faces: {len(faces)}", (10, 60),
                   config.FONT_FACE, config.FONT_SCALE,
                   config.COLOR_TEXT, config.FONT_THICKNESS)
        return annotated, faces

    def run(self):
        print("Press 'q' to quit, 's' to screenshot.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame, faces = self.process_frame(frame)
            cv2.imshow('Face Detection Demo', processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                filepath = os.path.join(self.screenshot_dir, filename)
                cv2.imwrite(filepath, processed_frame)
                print(f"Screenshot saved: {filepath}")
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        demo = WebcamDemo()
        demo.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)