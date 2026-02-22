import torch
import torch.nn as nn
from torchvision import models

class CNNLSTM(nn.Module):
    """Patch-sequence LSTM for a single image."""
    def __init__(self, hidden=256, num_layers=1):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])  # Bx512xHxW
        self.proj = nn.Linear(512, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        fm = self.cnn(x)  # B,512,H,W
        B, C, H, W = fm.shape
        tokens = fm.permute(0, 2, 3, 1).reshape(B, H*W, C)  # B,T,C
        tokens = self.proj(tokens)  # B,T,hidden
        out, _ = self.lstm(tokens)  # B,T,hidden
        last = out[:, -1, :]        # B,hidden
        logits = self.head(last)    # B,1
        return logits
