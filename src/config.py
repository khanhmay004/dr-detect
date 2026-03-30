"""
Configuration module for the Uncertainty-Aware Attention CNN pipeline.

Centralizes all hyperparameters, paths, and constants. Every other module
imports from here — never hard-codes magic numbers.
"""

import random
from pathlib import Path

import numpy as np
import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

# Dataset paths
APTOS_DIR = PROJECT_ROOT / "aptos" / "aptos2019-blindness-detection"
MESSIDOR_DIR = PROJECT_ROOT / "messidor-2"

APTOS_TRAIN_CSV = APTOS_DIR / "train.csv"
APTOS_TRAIN_IMAGES = APTOS_DIR / "train_images"
MESSIDOR_IMAGES = MESSIDOR_DIR / "IMAGES"
MESSIDOR_CSV = MESSIDOR_DIR / "messidor-2.csv"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
APTOS_PROCESSED_DIR = PROCESSED_DIR / "aptos"
MESSIDOR_PROCESSED_DIR = PROCESSED_DIR / "messidor2"
USE_PREPROCESSED_CACHE = True                # load from data/processed/ if available

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

# =============================================================================
# IMAGE PREPROCESSING CONFIGURATION
# =============================================================================

IMAGE_SIZE = 512
CROP_MARGIN = 0.05                          # extra margin around detected retina
BEN_GRAHAM_SIGMA_SCALE = 30                 # σ = image_width / this value

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

DR_GRADES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

REFERABLE_THRESHOLD = 2                     # grades >= 2 = referable DR
RANDOM_SEED = 42
N_FOLDS = 5
VALIDATION_FOLD = 0

# =============================================================================
# DATALOADER CONFIGURATION
# =============================================================================

BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

NUM_CLASSES = 5
CBAM_REDUCTION_RATIO = 16                   # channel attention bottleneck

# MC Dropout — Bayesian approximation
MC_DROPOUT_RATE = 0.5
MC_INFERENCE_PASSES = 20                    # T forward passes for uncertainty

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_T_MAX = EPOCHS
EARLY_STOPPING_PATIENCE = 5

# Focal Loss
FOCAL_GAMMA = 2.0                           # focusing parameter
FOCAL_ALPHA = None                          # None = auto-compute from class freq

# Gradient clipping
GRAD_CLIP_NORM = 1.0                        # max L2 norm for gradient clipping

# Mixed-precision training
USE_AMP = True                              # halves VRAM for 512×512 inputs

# =============================================================================
# EVALUATION & VISUALIZATION
# =============================================================================

BINARY_THRESHOLD = 0.5

DR_COLORS = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
    4: "#8e44ad",
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Fix all random seeds for full reproducibility.

    Also sets CuDNN to deterministic mode (small perf cost, but
    guarantees bitwise-identical results across runs).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_directories() -> None:
    """Create all necessary output directories."""
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
