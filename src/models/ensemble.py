import torch
import torch.nn as nn
from src.config import settings
from .model_factory import build_effnet_b0, build_resnet50, load_weights
from .cnn_lstm import CNNLSTM

class EnsembleModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.model_a = build_effnet_b0().to(device)
        self.model_b = build_resnet50().to(device)
        self.model_c = CNNLSTM().to(device)

        if settings.MODEL_A_PATH.exists():
            load_weights(self.model_a, settings.MODEL_A_PATH, device)
        if settings.MODEL_B_PATH.exists():
            load_weights(self.model_b, settings.MODEL_B_PATH, device)
        if settings.MODEL_C_PATH.exists():
            load_weights(self.model_c, settings.MODEL_C_PATH, device)

        self.primary_model = self.model_a

    def eval(self):
        self.model_a.eval(); self.model_b.eval(); self.model_c.eval()
        return super().eval()

    @torch.no_grad()
    def predict_proba(self, x):
        la = self.model_a(x).squeeze(1)
        lb = self.model_b(x).squeeze(1)
        lc = self.model_c(x).squeeze(1)

        pa = torch.sigmoid(la)
        pb = torch.sigmoid(lb)
        pc = torch.sigmoid(lc)

        p_final = (pa + pb + pc) / 3.0
        return {
            "model_a": float(pa.item()),
            "model_b": float(pb.item()),
            "model_c": float(pc.item()),
            "final": float(p_final.item()),
        }
