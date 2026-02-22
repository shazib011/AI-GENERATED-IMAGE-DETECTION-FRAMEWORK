import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from src.config import settings
from src.preprocessing.transforms import get_val_transform
from src.training.dataset import BinaryFolderDataset
from src.models.ensemble import EnsembleModel

def evaluate_ensemble(split="test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnsembleModel(device=device).eval()

    ds = BinaryFolderDataset(settings.DATA_DIR / split, transform=get_val_transform())
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            for i in range(x.size(0)):
                pi = model.predict_proba(x[i:i+1])["final"]
                y_prob.append(pi)
            y_true.extend(y.numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    f1  = f1_score(y_true, y_pred, zero_division=0)
    pr  = precision_score(y_true, y_pred, zero_division=0)
    rc  = recall_score(y_true, y_pred, zero_division=0)

    settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(settings.OUTPUTS_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    if not np.isnan(auc):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title(f"ROC Curve (AUC={auc:.3f})")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(settings.OUTPUTS_DIR / "roc_curve.png", bbox_inches="tight")
        plt.close()

    print({"AUC": auc, "F1": f1, "Precision": pr, "Recall": rc, "CM": cm.tolist()})

if __name__ == "__main__":
    evaluate_ensemble("test")
