from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"

# Active static classes (fist and peace removed)
CLASSES = [
    "thumbs_up",
    "thumbs_down",
    "open_palm",
    "heart",
    "snap",
    "no_gesture",
]

NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = MODELS_DIR / "resnet18_gesture.pth"
