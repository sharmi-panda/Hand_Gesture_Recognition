import torch.nn as nn
from torchvision import models
import config


def build_resnet_gesture_classifier(freeze_backbone: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, config.NUM_CLASSES)
    )

    return model
