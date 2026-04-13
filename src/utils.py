import json
import numpy as np
from pathlib import Path
from collections import Counter
from dataset import PlantDiseaseDataset


def print_dataset_summary(root_dir: str):
    ds = PlantDiseaseDataset(root_dir)

    print(f"\n{'='*50}")
    print(f"  Dataset : {root_dir}")
    print(f"  Classes : {len(ds.classes)}")
    print(f"  Samples : {len(ds.samples)}")
    print(f"{'='*50}")

    label_counts = Counter(label for _, label in ds.samples)

    print(f"\n{'Class':<45} {'Count':>6}")
    print("-" * 53)
    for cls, idx in ds.class_to_idx.items():
        print(f"{cls:<45} {label_counts[idx]:>6}")

    counts = list(label_counts.values())
    print(f"\nMin samples in a class : {min(counts)}")
    print(f"Max samples in a class : {max(counts)}")
    print(f"Imbalance ratio        : {max(counts)/min(counts):.1f}x")

    return ds


def compute_class_weights(ds: PlantDiseaseDataset) -> list:
    """
    Inverse frequency weighting.
    Classes with fewer samples get higher weight so the model pays more
    attention to them during training.
    """
    label_counts = Counter(label for _, label in ds.samples)
    total = len(ds.samples)
    n_classes = len(ds.classes)

    weights = []
    for idx in range(n_classes):
        count = label_counts.get(idx, 1)  # avoid division by zero
        weight = total / (n_classes * count)
        weights.append(round(weight, 4))

    return weights


def check_class_alignment(train_ds: PlantDiseaseDataset, val_ds: PlantDiseaseDataset):
    """
    Checks which classes are in train but not val, and vice versa.
    Important for understanding evaluation gaps.
    """
    train_classes = set(train_ds.classes)
    val_classes   = set(val_ds.classes)

    only_in_train = train_classes - val_classes
    only_in_val   = val_classes - train_classes

    print(f"\n{'='*50}")
    print(f"  Class alignment check")
    print(f"{'='*50}")
    print(f"  In train only ({len(only_in_train)} classes):")
    for c in sorted(only_in_train):
        print(f"    - {c}")

    if only_in_val:
        print(f"\n  In val only ({len(only_in_val)} classes):")
        for c in sorted(only_in_val):
            print(f"    - {c}")
    else:
        print(f"\n  No classes appear in val only. Good.")


def save_class_map(ds: PlantDiseaseDataset, output_path: str):
    """
    Saves class_to_idx and idx_to_class as JSON.
    The API will load this at startup to decode model outputs.
    """
    data = {
        "class_to_idx": ds.class_to_idx,
        "idx_to_class": {str(v): k for k, v in ds.class_to_idx.items()}
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Class map saved to: {output_path}")


if __name__ == "__main__":
    train_ds = print_dataset_summary("../data/train")
    val_ds   = print_dataset_summary("../data/val")

    check_class_alignment(train_ds, val_ds)

    weights = compute_class_weights(train_ds)
    print(f"\n  Class weights computed (first 5): {weights[:5]}")
    print(f"  Min weight: {min(weights):.4f}  Max weight: {max(weights):.4f}")

    save_class_map(train_ds, "../configs/class_map.json")
    print("\n  Phase 1 complete. Ready for training.")