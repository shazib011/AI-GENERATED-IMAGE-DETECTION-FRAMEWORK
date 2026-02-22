import torch
import torch.nn as nn
from torchvision import models

def build_effnet_b0(num_classes=1):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, num_classes)
    return m

def build_resnet50(num_classes=1):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m

def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model
