# DR-Detect Codebase Research Report

## Executive Summary

**DR-Detect** is an **Uncertainty-Aware Diabetic Retinopathy Detection** system built with PyTorch. It implements a deep learning pipeline to classify retinal fundus images into 5 DR severity grades (0-4). The project combines state-of-the-art techniques including CBAM attention, Monte-Carlo Dropout for Bayesian uncertainty, and Focal Loss for class imbalance handling.

**Purpose**: Develop a "Safety AI" system for:
1. 5-class DR grading (No DR, Mild, Moderate, Severe, Proliferative DR)
2. Binary referable DR detection (Grade >= 2)
3. Uncertainty quantification to flag uncertain cases for human review

---

## 1. Project Structure

```
c:\Projects\dr-detect\
├── src/                              # Main source code
│   ├── __init__.py                   # Package initializer
│   ├── config.py                     # Centralized configuration
│   ├── dataset.py                    # PyTorch Datasets & DataLoaders
│   ├── evaluate.py                   # MC Dropout inference & evaluation
│   ├── loss.py                       # Focal Loss implementation
│   ├── model.py                      # CBAMResNet50 architecture
│   ├── preprocessing.py              # Ben Graham preprocessing
│   └── train.py                      # Training script with AMP
├── aptos/                            # APTOS 2019 dataset
│   └── aptos2019-blindness-detection/
│       ├── train.csv                 # Training labels
│       ├── train_images/             # Training images (~3.6K)
│       ├── test.csv                  # Test labels
│       └── test_images/              # Test images
├── messidor-2/                       # External validation dataset
│   ├── IMAGES/                       # Retinal images (~1.7K)
│   └── messidor-2.csv                # Labels
├── outputs/
│   ├── checkpoints/                  # Saved model weights
│   ├── figures/                      # Visualization plots
│   ├── logs/                         # Training history JSON
│   └── results/                      # Evaluation metrics
├── notebooks/                        # Jupyter notebooks
│   └── 01_eda.ipynb                  # Exploratory Data Analysis
├── base_run/                         # Alternative experiments
├── docs/                             # Documentation
├── vast/                             # Vast.ai deployment scripts
├── dr-env/                           # Python virtual environment
└── requirements.txt                  # Dependencies
```

---

## 2. Core Architecture

### 2.1 Model: CBAMResNet50

```
Input (B, 3, 512, 512)
         │
         ├── ResNet-50 stem (conv1 + bn1 + relu + maxpool)
         ├── layer1 → CBAM(256)
         ├── layer2 → CBAM(512)
         ├── layer3 → CBAM(1024)
         ├── layer4 → CBAM(2048)
         │
         ├── AdaptiveAvgPool2d → (B, 2048)
         ├── MCDropout(p=0.5)    ← Always active (even in eval)
         └── Linear(2048, 5)
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| **ResNet-50** | Pretrained backbone (ImageNet1K_V2 weights) |
| **CBAM** | Convolutional Block Attention Module (channel + spatial attention) |
| **MCDropout** | Always-on dropout for Bayesian uncertainty quantification |

### 2.2 CBAM (Convolutional Block Attention Module)

**ChannelAttention:**
- Applies avg-pool AND max-pool over spatial dimensions
- Shared MLP: C → C/16 → C
- Sigmoid gating multiplication

**SpatialAttention:**
- Computes avg/max across channel dimension
- 7×7 convolution to produce spatial gate
- Sigmoid gating multiplication

**Design Decision:** CBAM placed AFTER each ResNet stage (not inside residual blocks) to maintain ImageNet pretrained weight compatibility.

### 2.3 MCDropout (Monte-Carlo Dropout)

```python
class MCDropout(nn.Module):
    def forward(self, x):
        return F.dropout(x, self.p, training=True)  # Always active!
```

**Purpose:** Enables Bayesian uncertainty quantification by keeping dropout active during inference. Multiple stochastic forward passes sample from an approximate posterior.

---

## 3. Data Pipeline

### 3.1 Datasets

| Dataset | Size | Purpose | Label Column |
|---------|------|---------|--------------|
| **APTOS 2019** | ~3,662 images | Training/Validation | `diagnosis` (0-4) |
| **Messidor-2** | ~1,748 images | External Validation | `adjudicated_dr_grade` |

**Class Distribution (APTOS):**
- Grade 0 (No DR): 49.3%
- Grade 1 (Mild): 10.1%
- Grade 2 (Moderate): 27.3%
- Grade 3 (Severe): 5.3%
- Grade 4 (Proliferative): 8.1%
- **Imbalance Ratio:** 9.4:1 (justifies Focal Loss)

### 3.2 Ben Graham Preprocessing

**Source:** Kaggle 2015 DR Competition winner technique

**Pipeline:**
1. **Circular Crop:** Detect retina boundary via thresholding + contour detection, crop with 5% margin
2. **Local Color Normalization:** `result = 4*image - 4*GaussianBlur(image) + 128`

**Purpose:** Removes black borders and illumination variation while preserving lesion detail.

### 3.3 Data Augmentation (Training)

```python
# Albumentations pipeline
HorizontalFlip(p=0.5)
VerticalFlip(p=0.5)
RandomRotate90(p=0.5)
ShiftScaleRotate(shift=0.1, scale=0.1, rotate=45)
OneOf([RandomBrightnessContrast, CLAHE], p=0.3)
GaussNoise(var_limit=(10, 50), p=0.2)
Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
```

### 3.4 Data Split

- **Method:** Stratified 5-Fold Cross-Validation
- **Preserves:** Class distribution across folds
- **Seed:** 42 (reproducible)

---

## 4. Loss Function: Focal Loss

**Formula:**
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

**Parameters:**
- `gamma = 2.0` — Down-weights easy/well-classified examples
- `alpha` — Inverse-frequency class weights (auto-computed)

**Purpose:** Addresses severe class imbalance where >50% are Grade 0 and <5% are Grade 4.

---

## 5. Training Pipeline

### 5.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 512×512 |
| Batch Size | 16 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Epochs | 20 |
| Early Stopping | 5 epochs patience |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| AMP | Enabled (mixed-precision) |

### 5.2 Training Features

- **Mixed-Precision (AMP):** 50% VRAM reduction, enables batch_size=16 on 12GB GPU
- **Early Stopping:** Tracks Quadratic Weighted Kappa
- **Checkpointing:** Best model + last checkpoint saved
- **Resume Support:** Can continue from checkpoint

### 5.3 Usage

```bash
python train.py                         # Default settings
python train.py --epochs 30 --lr 3e-4   # Custom hyperparameters
python train.py --resume checkpoint.pth # Resume training
python train.py --fold 0                # Specify validation fold
```

---

## 6. Uncertainty Quantification

### 6.1 MC Dropout Inference

**Process:**
1. Perform T=20 stochastic forward passes (dropout active)
2. Stack softmax probabilities: `(N, T, C)`
3. Compute predictive mean: `p̄(y|x) = 1/T × Σ p_t(y|x)`
4. Compute predictive entropy: `H[p̄] = -Σ p̄_c × log(p̄_c)`

**High entropy = Model uncertain → Flag for human review**

### 6.2 Evaluation Usage

```bash
python evaluate.py --checkpoint best.pth
python evaluate.py --checkpoint best.pth --mc_passes 50
```

### 6.3 Output Artifacts

- `messidor2_uncertainty.csv` — Per-image predictions + uncertainty
- `messidor2_metrics.json` — Aggregate metrics
- Visualization plots in `outputs/figures/`

---

## 7. Evaluation Metrics

### 7.1 Primary Metrics

| Metric | Purpose |
|--------|---------|
| **Quadratic Weighted Kappa (QWK)** | Ordinal classification (DR grades are ordered) |
| **AUC-ROC** | Binary referable DR (grade ≥ 2) |
| **Sensitivity/Specificity** | Clinical screening performance |

### 7.2 Uncertainty/Safety Metrics

| Metric | Purpose |
|--------|---------|
| **Predictive Entropy** | Uncertainty quantification |
| **Negative Log Likelihood (NLL)** | Calibration quality |
| **Expected Calibration Error (ECE)** | Confidence calibration |

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **512×512 resolution** | DR lesions are tiny; higher resolution preserves detail |
| **CBAM after stages** | Maintains ImageNet pretrained weight compatibility |
| **MCDropout always-on** | Enables uncertainty quantification in eval mode |
| **Focal Loss** | Handles severe class imbalance (medical datasets) |
| **AMP training** | Enables larger batches on consumer GPUs |
| **Ben Graham preprocessing** | Proven technique from Kaggle competition |
| **Single-layer classifier** | Avoids overfitting on small APTOS dataset |

---

## 9. Utility Functions

### 9.1 Configuration (`config.py`)

```python
seed_everything(seed=42)      # Full reproducibility
setup_directories()           # Create output directories
```

### 9.2 Loss Utilities (`loss.py`)

```python
compute_class_weights(labels, num_classes=5)  # Inverse-frequency weights
```

### 9.3 Data Utilities (`dataset.py`)

```python
get_train_val_split(df, val_fold=0, n_folds=5)
get_train_transform(image_size=512)
get_val_transform(image_size=512)
create_dataloaders(train_df, val_df, image_dir, ...)
create_messidor_dataloader(...)
```

### 9.4 Model Factory (`model.py`)

```python
create_model(num_classes=5, pretrained=True)  # Returns CBAMResNet50
```

---

## 10. Dependencies

| Category | Packages |
|----------|----------|
| **Core ML** | torch>=2.0.0, torchvision>=0.15.0 |
| **Image Processing** | opencv-python>=4.8.0, albumentations>=1.3.0 |
| **Data Science** | numpy>=1.24.0, pandas>=2.0.0, scikit-learn>=1.3.0 |
| **Visualization** | matplotlib>=3.7.0, seaborn>=0.12.0 |
| **Utilities** | tqdm>=4.65.0, PyYAML>=6.0 |
| **Jupyter** | jupyter>=1.0.0, ipykernel>=6.0.0 |

---

## 11. Cloud Deployment (Vast.ai)

Scripts in `vast/`:
- `setup_vastai.sh` — Instance setup
- `run_training.sh` — Training launcher
- `prepare_upload.py` — Data compression

**Expected Performance (RTX 3060/3070):**
- ~10-15 minutes per epoch
- ~3-5 hours for 20 epochs
- Cost: ~$1-2 per fold

---

## 12. DR Grade Reference

| Grade | Name | Color Code | Clinical Meaning |
|-------|------|------------|------------------|
| 0 | No DR | #2ecc71 (Green) | Healthy retina |
| 1 | Mild | #f1c40f (Yellow) | Microaneurysms only |
| 2 | Moderate | #e67e22 (Orange) | More than mild, less than severe |
| 3 | Severe | #e74c3c (Red) | Extensive abnormalities |
| 4 | Proliferative DR | #8e44ad (Purple) | Neovascularization, highest risk |

**Referable DR:** Grades 2-4 require specialist referral.

---

## 13. Code Patterns & Conventions

- **Constants:** `UPPER_SNAKE_CASE`
- **Classes:** `PascalCase`
- **Functions:** `snake_case`
- **Private methods:** `_prefixed`
- **Factory Pattern:** `create_model()`, `create_dataloaders()`
- **Trainer Class:** Encapsulates training loop, validation, checkpointing

---

## 14. Testing Approach

No formal test suite, but inline smoke tests exist:

```bash
python -m src.loss      # Gradient sanity check
python -m src.model     # Shape test + MC Dropout verification
python -m src.preprocessing image.jpg  # Single image test
```

---

## 15. Summary

DR-Detect is a well-architected medical imaging project implementing:

1. **Attention-enhanced CNN** (CBAM + ResNet-50)
2. **Bayesian uncertainty** via MC Dropout
3. **Class imbalance handling** via Focal Loss
4. **Reproducible training** with seeds, AMP, checkpointing
5. **External validation** on Messidor-2 dataset

The system is designed for clinical decision support where flagging uncertain predictions for human review is critical for patient safety.
