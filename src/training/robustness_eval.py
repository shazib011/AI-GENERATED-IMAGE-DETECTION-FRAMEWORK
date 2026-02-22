import io
import numpy as np
import torch
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from src.config import settings
from src.models.ensemble import EnsembleModel
from src.preprocessing.transforms import get_infer_transform
from src.preprocessing.align_crop import crop_face_or_full
from src.training.dataset import BinaryFolderDataset

def jpeg_attack(pil_img, quality=50):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

def blur_attack(pil_img, radius=1.0):
    return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

def noise_attack(pil_img, sigma=10):
    arr = np.array(pil_img).astype(np.float32)
    arr += np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnsembleModel(device=device).eval()
    tfm = get_infer_transform()

    ds = BinaryFolderDataset(settings.DATA_DIR / "test", transform=None)

    attacks = [
        ("clean", None),
        ("jpeg_50", lambda im: jpeg_attack(im, 50)),
        ("jpeg_30", lambda im: jpeg_attack(im, 30)),
        ("blur_1",  lambda im: blur_attack(im, 1.0)),
        ("noise_10",lambda im: noise_attack(im, 10)),
    ]

    results = {}
    N = min(300, len(ds.samples))
    for name, atk in attacks:
        correct = 0
        for fp, y in ds.samples[:N]:
            im = Image.open(fp).convert("RGB")
            im, _ = crop_face_or_full(im)
            if atk:
                im = atk(im)
            x = tfm(im).unsqueeze(0).to(device)
            p = model.predict_proba(x)["final"]
            pred = 1 if p >= 0.5 else 0
            correct += int(pred == y)
        acc = correct / max(1, N)
        results[name] = acc
        print(name, acc)

    settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(list(results.keys()), list(results.values()))
    plt.xticks(rotation=30, ha="right")
    plt.title("Robustness Accuracy")
    plt.savefig(settings.OUTPUTS_DIR / "robustness.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run()
