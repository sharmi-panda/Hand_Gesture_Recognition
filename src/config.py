# src/config.py
from pathlib import Path
import torch

# Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"

# Ensure runtime directories exist
for folder in [RAW_DATA_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Gesture Target Classes
CLASSES = ["thumbs_up", "thumbs_down", "open_palm", "fist", "peace", "no_gesture"]
NUM_CLASSES = len(CLASSES)

# Model & Training Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 12
LEARNING_RATE = 1e-3

# Model Weight Checkpoint Location
MODEL_PATH = MODELS_DIR / "resnet18_gesture.pth"

# Inference Engine Device (Intel CPU Default)
DEVICE = torch.device("cpu")