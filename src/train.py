import time
import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_data_loaders
from model import build_resnet_gesture_classifier


def train():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_data_loaders()

    model = build_resnet_gesture_classifier(freeze_backbone=False).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_acc = 0.0
    print(f"\n[INFO] Training ResNet-18 on {config.DEVICE} for {config.NUM_EPOCHS} epochs...\n")

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct_train += preds.eq(labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct_val += preds.eq(labels).sum().item()
                total_val += labels.size(0)

        val_loss /= total_val
        val_acc = (correct_val / total_val) * 100.0
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch+1:02d}/{config.NUM_EPOCHS:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  --> Saved new best checkpoint to {config.MODEL_PATH}")

    print(f"\n[COMPLETE] Training finished. Best Validation Accuracy: {best_val_acc:.2f}%\n")


if __name__ == "__main__":
    train()
