import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def make_gradcam_overlay(pil_img, model, x_tensor):
    model.eval()
    grads = {}
    feats = {}

    target_layer = model.features[-1]  # EfficientNet

    def fwd_hook(_, __, output):
        feats["value"] = output

    def bwd_hook(_, grad_in, grad_out):
        grads["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x_tensor = x_tensor.requires_grad_(True)
    logits = model(x_tensor).squeeze(1)
    score = logits[0]
    model.zero_grad()
    score.backward()

    f = feats["value"]
    g = grads["value"]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = (w * f).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove(); h2.remove()

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_img.size)
    cam_np = np.array(cam_img).astype(np.float32) / 255.0
    img_np = np.array(pil_img).astype(np.float32) / 255.0

    overlay = img_np.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.6 * cam_np, 0, 1)

    out = Image.fromarray((overlay * 255).astype(np.uint8))
    return out
