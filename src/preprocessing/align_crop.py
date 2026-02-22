from PIL import Image
from .face_detector import FaceDetector

_detector = FaceDetector()

def crop_face_or_full(pil_img: Image.Image, expand: float = 0.25):
    box = _detector.detect(pil_img)
    if box is None:
        return pil_img, False

    x, y, w, h = box
    W, H = pil_img.size
    pad_w, pad_h = int(w * expand), int(h * expand)

    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(W, x + w + pad_w)
    y1 = min(H, y + h + pad_h)

    return pil_img.crop((x0, y0, x1, y1)), True
