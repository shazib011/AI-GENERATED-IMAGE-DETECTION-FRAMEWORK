import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import settings
from src.preprocessing.transforms import get_train_transform, get_val_transform
from src.training.dataset import BinaryFolderDataset
from src.models.model_factory import build_effnet_b0, build_resnet50
from src.models.cnn_lstm import CNNLSTM

def build_model(name: str):
    name = name.lower()
    if name == "effnet":
        return build_effnet_b0()
    if name == "resnet":
        return build_resnet50()
    if name == "cnnlstm":
        return CNNLSTM()
    raise ValueError("model name must be: effnet | resnet | cnnlstm")

def evaluate_acc(model, dl, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            p = torch.sigmoid(model(x).squeeze(1))
            pred = (p >= 0.5).long()
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)

def train(model_name="effnet", epochs=5, batch_size=16, lr=1e-4, out_path="weights/model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name).to(device)

    train_ds = BinaryFolderDataset(settings.DATA_DIR / "train", transform=get_train_transform())
    val_ds   = BinaryFolderDataset(settings.DATA_DIR / "val", transform=get_val_transform())

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    best_val = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Train {model_name} ep{ep}/{epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.float().to(device)

            logits = model(x).squeeze(1)
            loss = crit(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        acc = evaluate_acc(model, val_dl, device)
        print(f"[VAL] acc={acc:.4f}")

        if acc > best_val:
            best_val = acc
            torch.save({"state_dict": model.state_dict()}, out_path)
            print(f"Saved best to {out_path}")

if __name__ == "__main__":
    # Run these one by one (change model_name and out_path):
    # 1) effnet  -> weights/model_a_effnet.pth
    # 2) resnet  -> weights/model_b_resnet.pth
    # 3) cnnlstm -> weights/model_c_cnnlstm.pth

    # Example (Model A):
   train(model_name="effnet", epochs=5, out_path="weights/model_a_effnet.pth")





