import io, base64
from PIL import Image
import torch
from src.config import settings
from src.preprocessing.align_crop import crop_face_or_full
from src.preprocessing.transforms import get_infer_transform
from src.models.ensemble import EnsembleModel
from src.models.gradcam import make_gradcam_overlay
from src.utils.logger import log_prediction, now_iso
from src.utils.timing import timer_ms

class Predictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = get_infer_transform()
        self.model = EnsembleModel(device=self.device).eval()
        self.models_loaded = True

    def predict_bytes(self, img_bytes: bytes, with_heatmap: bool = True):
        with timer_ms() as ms:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            face_img, face_found = crop_face_or_full(img)
            x = self.transform(face_img).unsqueeze(0).to(self.device)

            probs = self.model.predict_proba(x)
            p = probs["final"]

            label = "FAKE" if p >= settings.THRESHOLD else "REAL"
            confidence = float(p if p >= settings.THRESHOLD else (1 - p))

            heatmap_b64 = None
            if with_heatmap and hasattr(self.model, "primary_model"):
                try:
                    overlay = make_gradcam_overlay(face_img, self.model.primary_model, x)
                    heatmap_b64 = pil_to_b64(overlay)
                except Exception:
                    heatmap_b64 = None

        row = {
            "time": now_iso(),
            "label": label,
            "confidence": round(confidence, 4),
            "prob_fake": round(float(p), 4),
            "face_found": int(face_found),
            "runtime_ms": ms(),
        }
        log_prediction(settings.LOGS_DIR / "predictions.csv", row)

        return {
            "label": label,
            "confidence": confidence,
            "prob_fake": float(p),
            "face_found": bool(face_found),
            "model_probs": probs,
            "heatmap": heatmap_b64,
            "runtime_ms": ms(),
        }

def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
