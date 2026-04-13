import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def get_transforms(img_size: int, mode: str):
    """
    Returns train or val transforms.
    mode: 'train' | 'val'
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomRotation(30),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 1.1)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


class PlantDiseaseDataset(Dataset):
    """
    Loads images from a folder where each subfolder = one disease class.
    e.g.
        data/train/apple black rot/img001.jpg
        data/train/bean rust/img002.jpg

    Returns (image_tensor, label_index).
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []   # list of (image_path, label_idx)

        # Build class list — sorted so index is always deterministic
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image paths
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __repr__(self):
        return (
            f"PlantDiseaseDataset(root='{self.root_dir}', "
            f"classes={len(self.classes)}, samples={len(self.samples)})"
        )