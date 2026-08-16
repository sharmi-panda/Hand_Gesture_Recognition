# src/train.py
import time
import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import build_dataloaders
from model import build_resnet_gesture_classifier


def train_model():
    print("=" * 60)
    print(" STARTING RESNET-18 HAND GESTURE TRAINING (CPU)")
    print("=" * 60)

    train_loader, val_loader = build_dataloaders()
    model = build_resnet_gesture_classifier(freeze_backbone=True).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=config.LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()

        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{config.NUM_EPOCHS:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.1f}% | "
            f"Val Acc: {val_acc*100:.1f}%"
        )

        # Save Best Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  └─> [SAVED] Best checkpoint updated -> {config.MODEL_PATH.name}")

    print("\n[COMPLETE] Training finished. Best Validation Accuracy:", f"{best_val_acc*100:.2f}%")


if __name__ == "__main__":
    train_model()