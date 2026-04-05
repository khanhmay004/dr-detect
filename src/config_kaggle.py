import random
from pathlib import Path

import numpy as np
import torch

KAGGLE_DATA_ROOT = Path("/kaggle/input/datasets/meiuwu04")

PROJECT_ROOT = Path("/kaggle/working/dr-detect")

APTOS_DIR = KAGGLE_DATA_ROOT / "data-split" / "data_split"
MESSIDOR_DIR = KAGGLE_DATA_ROOT / "messidor-2" / "messidor-2"

APTOS_TRAIN_CSV = APTOS_DIR / "train_label.csv"
APTOS_TRAIN_IMAGES = APTOS_DIR / "train_split"

APTOS_TEST_CSV = APTOS_DIR / "test_label.csv"
APTOS_TEST_IMAGES = APTOS_DIR / "test_split"

MESSIDOR_IMAGES = MESSIDOR_DIR / "IMAGES"
MESSIDOR_CSV = MESSIDOR_DIR / "messidor_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
APTOS_PROCESSED_DIR = PROCESSED_DIR / "aptos"
MESSIDOR_PROCESSED_DIR = PROCESSED_DIR / "messidor2"
USE_PREPROCESSED_CACHE = False

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

IMAGE_SIZE = 512
CROP_MARGIN = 0.05
BEN_GRAHAM_SIGMA_SCALE = 30

DR_GRADES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

REFERABLE_THRESHOLD = 2
RANDOM_SEED = 42
N_FOLDS = 5
VALIDATION_FOLD = 0

BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 5
CBAM_REDUCTION_RATIO = 16

MC_DROPOUT_RATE = 0.5
MC_INFERENCE_PASSES = 20

EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_T_MAX = EPOCHS
EARLY_STOPPING_PATIENCE = 5

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None

GRAD_CLIP_NORM = 1.0

USE_AMP = True

LABEL_SMOOTHING = 0.0
USE_BALANCED_SAMPLER = False
CLASSIFIER_HIDDEN_DIM = 0
CLASSIFIER_DROPOUT_RATE = MC_DROPOUT_RATE
LR_WARMUP_EPOCHS = 0

USE_AUG_BALANCED_DATASET: bool = False
AUG_TARGET_COUNT_PER_CLASS: int = 800
AUG_FOCAL_ALPHA_UNIFORM: bool = True
AUG_GRADE_LEVEL_OVERRIDE: dict[int, str] | None = None

BINARY_THRESHOLD = 0.5

DR_COLORS = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
    4: "#8e44ad",
}


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_directories() -> None:
    for dir_path in [OUTPUT_DIR, FIGURES_DIR, PROCESSED_DIR,
                     CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"APTOS Train Images: {APTOS_TRAIN_IMAGES}")
    print(f"Messidor-2 Images:  {MESSIDOR_IMAGES}")
    print(f"Image Size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    setup_directories()
    print("Directories created!")
