import cv2
import numpy as np
from PIL import Image

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, pil_img: Image.Image):
        img = np.array(pil_img)[:, :, ::-1]  # RGB->BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])  # largest face
        return (int(x), int(y), int(w), int(h))
