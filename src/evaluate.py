import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('src')
from dataset import PlantDiseaseDataset, get_transforms
from model import build_model


@torch.no_grad()
def compute_map(model, dataloader, num_classes: int, device: torch.device) -> dict:
    """
    Computes mean Average Precision (mAP) across all classes.
    This is the primary evaluation metric used by the hidden test set.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs)   # (N, num_classes)
    all_labels = np.concatenate(all_labels)  # (N,)

    # One-hot encode for AP computation
    one_hot = np.eye(num_classes)[all_labels]  # (N, num_classes)

    per_class_ap = []
    for c in range(num_classes):
        if one_hot[:, c].sum() == 0:
            # Class not present in val — skip
            continue
        ap = average_precision_score(one_hot[:, c], all_probs[:, c])
        per_class_ap.append(ap)

    return {
        "mAP"          : float(np.mean(per_class_ap)),
        "per_class_AP" : per_class_ap,
        "all_probs"    : all_probs,
        "all_labels"   : all_labels,
    }


def plot_confusion_matrix(all_labels, all_preds, class_names, output_path="reports/confusion_matrix.png"):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


if __name__ == "__main__":
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class map
    with open("configs/class_map.json") as f:
        class_map = json.load(f)
    idx_to_class  = class_map["idx_to_class"]
    class_to_idx  = class_map["class_to_idx"]
    num_classes   = len(idx_to_class)
    class_names   = [idx_to_class[str(i)] for i in range(num_classes)]

    # Load val dataset with train's mapping
    val_ds = PlantDiseaseDataset(
        "data/val",
        transform=get_transforms(300, "val"),
        class_to_idx=class_to_idx
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # Load model
    model = build_model("efficientnet_b3", num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load("weights/best_model.pt", map_location=device))
    model = model.to(device)

    # Compute mAP
    results = compute_map(model, val_loader, num_classes, device)
    print(f"\nmAP: {results['mAP']:.4f}")

    # Confusion matrix
    all_preds = results["all_probs"].argmax(axis=1)
    os.makedirs("reports", exist_ok=True)
    plot_confusion_matrix(results["all_labels"], all_preds, class_names)