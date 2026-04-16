
import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
import wandb

sys.path.append('/kaggle/working/src')
from dataset import PlantDiseaseDataset, get_transforms
from model import build_model, count_parameters


def compute_class_weights(dataset: PlantDiseaseDataset) -> torch.Tensor:
    label_counts = Counter(label for _, label in dataset.samples)
    total     = len(dataset.samples)
    n_classes = len(dataset.classes)
    weights   = [
        total / (n_classes * label_counts.get(i, 1))
        for i in range(n_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


def run_training(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build train dataset first to get master class mapping ──
    train_ds = PlantDiseaseDataset(
        config["train_dir"],
        transform=get_transforms(config["img_size"], "train")
    )

    # ── Val uses train's exact class_to_idx ────────────────────
    val_ds = PlantDiseaseDataset(
        config["val_dir"],
        transform=get_transforms(config["img_size"], "val"),
        class_to_idx=train_ds.class_to_idx  # <-- KEY FIX
    )

    print(f"Train: {len(train_ds)} samples, {len(train_ds.classes)} classes")
    print(f"Val  : {len(val_ds)} samples, {len(val_ds.classes)} classes (using train mapping)")

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True
    )

    # ── Model ──────────────────────────────────────────────────
    model = build_model(
        backbone    = config["backbone"],
        num_classes = len(train_ds.classes),
        pretrained  = True,
        dropout     = config["dropout"]
    )
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # ── Loss with class weighting ───────────────────────────────
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight          = class_weights,
        label_smoothing = config["label_smoothing"]
    )

    # ── Optimizer & scheduler ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    # ── Save class map ──────────────────────────────────────────
    os.makedirs("/kaggle/working/configs", exist_ok=True)
    class_map = {
        "class_to_idx": train_ds.class_to_idx,
        "idx_to_class": {str(v): k for k, v in train_ds.class_to_idx.items()}
    }
    with open("/kaggle/working/configs/class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)
    print("Class map saved.")

    # ── W&B ────────────────────────────────────────────────────
    wandb.init(project="plant-disease", config=config, name=config["run_name"])

    # ── Training loop ───────────────────────────────────────────
    best_val_acc = 0.0
    os.makedirs("/kaggle/working/weights", exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{config['epochs']} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
              f"LR: {lr:.6f}")

        wandb.log({
            "epoch"     : epoch,
            "train/loss": train_loss,
            "train/acc" : train_acc,
            "val/loss"  : val_loss,
            "val/acc"   : val_acc,
            "lr"        : lr,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                "/kaggle/working/weights/best_model.pt"
            )
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    wandb.finish()
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    return model, train_ds.classes
