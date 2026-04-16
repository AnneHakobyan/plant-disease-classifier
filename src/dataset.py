
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def get_transforms(img_size: int, mode: str):
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
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root_dir: str, transform=None, class_to_idx: dict = None):
        """
        Args:
            root_dir:     folder where each subfolder = one disease class
            transform:    image transforms
            class_to_idx: if provided, use this fixed mapping instead of
                          building one from root_dir. Pass train's mapping
                          to val so indices are always consistent.
        """
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []

        if class_to_idx is not None:
            # Use the provided mapping (val uses train's mapping)
            self.class_to_idx = class_to_idx
            self.classes      = sorted(class_to_idx.keys())
        else:
            # Build mapping from scratch (train does this)
            self.classes = sorted([
                d.name for d in self.root_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect samples — skip classes not in the mapping
        for cls_dir in self.root_dir.iterdir():
            if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                continue
            if cls_dir.name not in self.class_to_idx:
                continue  # val class not in train — skip
            label = self.class_to_idx[cls_dir.name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
