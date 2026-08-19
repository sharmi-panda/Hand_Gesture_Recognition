from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


class GestureDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_data_loaders(test_size: float = 0.2, random_state: int = 42):
    all_paths = []
    all_labels = []

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(config.CLASSES)}

    for cls_name, idx in class_to_idx.items():
        cls_dir = config.RAW_DATA_DIR / cls_name
        if not cls_dir.exists():
            continue
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        all_paths.extend(images)
        all_labels.extend([idx] * len(images))

    if not all_paths:
        raise FileNotFoundError(f"No images found under '{config.RAW_DATA_DIR}'. Run collect_data.py first.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )

    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = GestureDataset(train_paths, train_labels, transform=train_transform)
    val_ds = GestureDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader
