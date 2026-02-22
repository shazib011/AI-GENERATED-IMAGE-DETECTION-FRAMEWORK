from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class BinaryFolderDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.transform = transform
        self.samples = []
        for label_name, label in [("real", 0), ("fake", 1)]:
            p = root_dir / label_name
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for fp in p.rglob(ext):
                    self.samples.append((fp, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, label = self.samples[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
