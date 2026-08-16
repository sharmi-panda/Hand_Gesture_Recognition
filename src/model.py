# src/model.py
import torch
import torch.nn as nn
from torchvision import models
import config


def build_resnet_gesture_classifier(
    num_classes: int = config.NUM_CLASSES, 
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Instantiates ResNet-18 with ImageNet weights and replaces the output head.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Unfreeze higher layers and replace final classifier layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )

    return model


if __name__ == "__main__":
    net = build_resnet_gesture_classifier()
    dummy_input = torch.randn(1, 3, 224, 224)
    out = net(dummy_input)
    print(f"[MODEL CHECK] Output tensor shape: {out.shape}")