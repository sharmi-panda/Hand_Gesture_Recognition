# src/dataset.py
from pathlib import Path
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

import config


class GestureDataset(Dataset):
    """Custom PyTorch Dataset for loading cropped hand gesture images."""

    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Data augmentation transforms for training and standard normalization for validation."""
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def build_dataloaders(val_split: float = 0.20) -> Tuple[DataLoader, DataLoader]:
    """Scans raw data directory and creates stratified train and validation PyTorch DataLoaders."""
    paths, labels = [], []
    class_map = {name: i for i, name in enumerate(config.CLASSES)}

    for class_name in config.CLASSES:
        class_dir = config.RAW_DATA_DIR / class_name
        if not class_dir.exists():
            continue

        for file_path in class_dir.glob("*.jpg"):
            paths.append(file_path)
            labels.append(class_map[class_name])

    if not paths:
        raise FileNotFoundError(
            f"No images found in {config.RAW_DATA_DIR}. Run 'python src/collect_data.py' first!"
        )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=val_split, random_state=42, stratify=labels
    )

    train_tf, val_tf = get_transforms()

    train_ds = GestureDataset(train_paths, train_labels, transform=train_tf)
    val_ds = GestureDataset(val_paths, val_labels, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"[DATASET] Loaded {len(paths)} images | Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader