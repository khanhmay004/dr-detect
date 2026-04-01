# Data Preprocessing and Pipeline Analysis

> **Document Role**: Comprehensive analysis of the data preprocessing pipeline and data leakage concerns for the DR-Detect project.
> **Created**: 2026-04-01
> **Scope**: APTOS 2019 and Messidor-2 datasets

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Data Pipeline Architecture](#2-current-data-pipeline-architecture)
3. [APTOS Dataset Structure Analysis](#3-aptos-dataset-structure-analysis)
4. [Why `data/processed/aptos` Has No Train/Test Subfolders](#4-why-dataprocessedaptos-has-no-traintest-subfolders)
5. [Preprocessing Steps Documentation](#5-preprocessing-steps-documentation)
6. [Patient-Level Split Analysis and Leakage Concerns](#6-patient-level-split-analysis-and-leakage-concerns)
7. [Training and Evaluation Pipeline Verification](#7-training-and-evaluation-pipeline-verification)
8. [Identified Issues and Recommendations](#8-identified-issues-and-recommendations)
9. [Mathematical Analysis of Preprocessing Impact](#9-mathematical-analysis-of-preprocessing-impact)
10. [Summary of Findings](#10-summary-of-findings)

---

## 1. Executive Summary

This document provides a comprehensive analysis of the data preprocessing and pipeline architecture in the DR-Detect project. The analysis addresses three key concerns:

1. **Is the training/evaluation split correctly implemented?** — **YES**, the pipeline correctly uses APTOS train data with 5-fold cross-validation and Messidor-2 as external test set.

2. **Why doesn't `data/processed/aptos` have train/test subfolders?** — **By design**, the APTOS test set lacks ground-truth labels (Kaggle competition format), making it unusable for supervised learning. Only the labeled training data is preprocessed.

3. **Are preprocessing steps documented with patient-level splits verified?** — **PARTIALLY**. Preprocessing is well-documented (Ben Graham's pipeline), but patient-level splits are **structurally impossible** for APTOS (no patient IDs provided). This is a known limitation.

---

## 2. Current Data Pipeline Architecture

### 2.1. High-Level Data Flow

```
                         APTOS 2019 Dataset
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   train.csv              test.csv              sample_submission.csv
   (3,662 rows)          (1,928 rows)
   id_code + diagnosis   id_code ONLY (!)
        │                      │
        │                      └─────────────────────────────┐
        │                                                    │
   train_images/                                        test_images/
   (3,662 images)                                       (1,928 images)
        │                                                    │
        ▼                                               NOT USABLE
   preprocess_data.py                              (no ground-truth labels)
        │
        ▼
   data/processed/aptos/
   (3,662 preprocessed images, FLAT structure)
        │
        ├─────────────────────────────────┐
        │                                 │
   Stratified 5-Fold CV (seed=42)         │
        │                                 │
        ├── Fold 0: Train ~2,929          │
        │           Val ~733              │
        ├── Fold 1–4: Same 80/20 split    │
        │                                 │
        ▼                                 │
   APTOS Training & Validation            │
                                          │
                         ┌────────────────┘
                         │
                         ▼
                    Messidor-2 Dataset
                    (1,748 images, 1,745 gradable)
                         │
                         ▼
                   External Test Set
                   (Never used for training)
```

### 2.2. Key Files in the Pipeline

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `src/config.py` | Centralized paths and hyperparameters | — | Path constants |
| `src/preprocessing.py` | Ben Graham's preprocessing functions | Raw BGR image | Preprocessed BGR image |
| `src/preprocess_data.py` | Offline batch preprocessing | Raw dataset directories | `data/processed/` cache |
| `src/dataset.py` | PyTorch Dataset classes | CSV + images | `(tensor, label)` tuples |
| `src/train.py` | Training loop with 5-fold CV | DataLoaders | Checkpoints, metrics |
| `src/evaluate.py` | External validation on Messidor-2 | Checkpoint + Messidor-2 | Metrics, uncertainty |

---

## 3. APTOS Dataset Structure Analysis

### 3.1. Raw Dataset Contents

```
aptos/aptos2019-blindness-detection/
├── train.csv              # 3,663 lines (header + 3,662 data rows)
│   └── Columns: id_code, diagnosis (0-4)
├── test.csv               # 1,929 lines (header + 1,928 data rows)
│   └── Columns: id_code ONLY (no labels!)
├── sample_submission.csv  # Kaggle submission format
├── train_images/          # 3,662 PNG files
└── test_images/           # 1,928 PNG files
```

### 3.2. Critical Observation: Test Set Has No Labels

The APTOS dataset was released as part of a Kaggle competition. The test set labels were **never publicly released** — they remain on Kaggle's servers for scoring submissions.

**Evidence from `test.csv` header:**
```csv
id_code
0005cfc8afb6
003f0afdcd15
006efc72b638
00836aaacf06
```

Compare to `train.csv`:
```csv
id_code,diagnosis
000c1434d8d7,2
001639a390f0,4
0024cdab0c1e,1
002c21358ce6,0
```

**Implication**: The APTOS test set is **unusable for supervised learning**. Any model evaluation must use:
1. Cross-validation on the labeled training set, OR
2. An external dataset with known labels (Messidor-2)

### 3.3. Class Distribution (Training Set Only)

| Grade | Name             | Count | Percentage | Imbalance Ratio vs. Grade 3 |
|-------|------------------|-------|------------|------------------------------|
| 0     | No DR            | 1,805 | 49.3%      | 9.4:1                        |
| 1     | Mild NPDR        | 370   | 10.1%      | 1.9:1                        |
| 2     | Moderate NPDR    | 999   | 27.3%      | 5.2:1                        |
| 3     | Severe NPDR      | 193   | 5.3%       | 1.0:1 (reference)            |
| 4     | Proliferative DR | 295   | 8.1%       | 1.5:1                        |

This severe imbalance (9.4:1 between majority and minority classes) justifies the use of Focal Loss with inverse-frequency alpha weighting.

---

## 4. Why `data/processed/aptos` Has No Train/Test Subfolders

### 4.1. Design Decision Rationale

The `data/processed/aptos/` directory contains **all 3,662 preprocessed training images** in a flat structure without train/test subfolders. This is **correct behavior** for the following reasons:

1. **APTOS test labels are unavailable** — Cannot evaluate models on APTOS test set.
2. **Dynamic splitting via 5-fold CV** — Train/val splits are computed at runtime, not fixed on disk.
3. **Preprocessing is label-agnostic** — Ben Graham's pipeline transforms images identically regardless of their eventual use.
4. **Messidor-2 serves as external test** — Eliminates need for APTOS test set entirely.

### 4.2. Code Evidence

From `src/preprocess_data.py` (lines 417–438):
```python
# Process APTOS
if args.dataset in ["aptos", "all"]:
    aptos_df = pd.read_csv(APTOS_TRAIN_CSV)  # Only train.csv (has labels)
    aptos_output = PROCESSED_DIR / "aptos"    # Flat output directory

    stats = preprocess_dataset(
        df=aptos_df,
        source_dir=APTOS_TRAIN_IMAGES,        # Only train_images/
        output_dir=aptos_output,
        id_column="id_code",
        ...
    )
```

The script explicitly reads `APTOS_TRAIN_CSV` (which points to `train.csv`) and processes images from `APTOS_TRAIN_IMAGES` (which points to `train_images/`). The APTOS test set is **never touched**.

### 4.3. Alternative Design (Rejected)

One could preprocess the APTOS test images and store them in `data/processed/aptos/test/`, but this would serve no purpose:
- No ground-truth labels → Cannot compute metrics
- Cannot submit to Kaggle → Competition closed in 2019
- Wastes disk space and processing time

---

## 5. Preprocessing Steps Documentation

### 5.1. Overview of Implemented Steps

| Step | Method | Implemented | Location |
|------|--------|-------------|----------|
| **Circular Crop** | Contour detection + enclosing circle | ✅ Yes | `preprocessing.py:27–53` |
| **Local Color Normalization** | Ben Graham's Gaussian blur subtraction | ✅ Yes | `preprocessing.py:98–119` |
| **CLAHE** | Contrast-limited adaptive histogram equalization | ⚠️ Replaced | See §5.4 |
| **GAN Augmentation** | Synthetic image generation | ❌ Not implemented | Future work |
| **Super-Resolution** | Upscaling via neural networks | ❌ Not implemented | Not planned |

### 5.2. Step 1: Circular Crop

**Purpose**: Remove black borders and camera artifacts from fundus images.

**Algorithm**:
1. Convert to grayscale
2. Threshold at intensity 10 (separate retina from black surround)
3. Morphological closing to fill gaps
4. Find largest contour → minimum enclosing circle
5. Add 5% margin to avoid edge clipping
6. Crop to bounding box, resize to 512×512
7. Apply circular mask (zero-out corners)

**Code Reference** (`preprocessing.py:55–91`):
```python
def circular_crop(
    image: np.ndarray,
    target_size: int = IMAGE_SIZE,
    margin: float = CROP_MARGIN,
) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy, r = find_retina_circle(image)
    r = int(r * (1 + margin))  # Add margin

    # Crop and resize
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (target_size, target_size),
                         interpolation=cv2.INTER_LANCZOS4)

    # Apply circular mask
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    cv2.circle(mask, (center, center), center, 255, -1)
    resized[mask == 0] = 0

    return resized
```

**Mathematical Impact**:
- Removes useless background pixels → Reduces noise in feature extraction
- Standardizes image dimensions → Consistent input to CNN
- Lanczos4 interpolation preserves fine detail (important for microaneurysms)

### 5.3. Step 2: Local Color Normalization (Ben Graham's Method)

**Purpose**: Remove illumination variation while preserving lesion detail.

**Formula**:
$$
\text{result} = 4 \cdot \text{image} - 4 \cdot \text{GaussianBlur}(\text{image}, \sigma) + 128
$$

Where $\sigma = \frac{\text{image\_width}}{30}$ (adaptive to image resolution).

**Intuition**:
- The local average (Gaussian blur) captures the illumination gradient
- Subtracting it removes slow-varying lighting changes
- The factor of 4 amplifies remaining contrast
- Adding 128 re-centers pixel values to mid-gray (prevents clipping)

**Code Reference** (`preprocessing.py:98–119`):
```python
def local_color_normalization(
    image: np.ndarray,
    sigma_scale: int = BEN_GRAHAM_SIGMA_SCALE,  # 30
) -> np.ndarray:
    h, w = image.shape[:2]
    sigma = w / sigma_scale  # Adaptive sigma

    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    result = cv2.addWeighted(image, 4, blurred, -4, 128)

    return result
```

**Key Properties**:
- Operates on all 3 BGR channels simultaneously (preserves color relationships)
- High-pass filter effect: suppresses low-frequency (lighting), preserves high-frequency (lesions)
- Deterministic: same input always produces same output

### 5.4. CLAHE: Replaced, Not Used

**Important Clarification**: The project documentation mentions CLAHE in training augmentations, but this is **distinct from preprocessing**.

| Context | CLAHE Usage | Purpose |
|---------|-------------|---------|
| **Offline Preprocessing** | ❌ NOT USED | Ben Graham's method replaces CLAHE |
| **Training Augmentation** | ✅ Used (p=0.3 in OneOf) | Runtime photometric variation |

**Code Evidence** (`dataset.py:305–310`):
```python
A.OneOf([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    A.CLAHE(clip_limit=2.0, p=1.0),  # Only in augmentation, not preprocessing
], p=0.3),
```

**Rationale for Replacing CLAHE with Ben Graham's Method**:

1. **CLAHE operates per-tile, per-channel** — Can introduce color artifacts
2. **Ben Graham's method operates globally** — Preserves color relationships
3. **Winning Kaggle solution** — Empirically proven on DR detection
4. **Simpler implementation** — Single line vs. multiple parameters

### 5.5. GAN Augmentation: Not Implemented

**Status**: Not implemented in current pipeline.

**Potential Approaches** (from literature):
- StyleGAN-based fundus synthesis (Kim et al. 2022)
- Conditional cascaded diffusion models (CCDM, 2025)
- Lesion-specific synthesis for minority classes

**Why Not Implemented**:
1. Complexity overhead for bachelor's thesis scope
2. Requires separate training pipeline
3. Risk of introducing synthetic artifacts
4. Standard augmentations sufficient for proof-of-concept

**Recommendation**: Consider for future work if minority-class performance (Grades 3-4) remains low.

### 5.6. Super-Resolution: Not Planned

**Status**: Not implemented, not planned.

**Rationale**:
- Images already resized to 512×512 (sufficient for lesion detection)
- ResNet-50 pre-trained on 224×224 ImageNet (upscaling provides diminishing returns)
- Super-resolution adds inference latency and complexity
- Literature shows marginal gains for DR (Ma et al. 2023)

---

## 6. Patient-Level Split Analysis and Leakage Concerns

### 6.1. The Patient-Level Split Problem

**Definition**: Data leakage occurs when information from the test set inadvertently influences training. In medical imaging, this commonly happens when:
1. Multiple images from the same patient appear in both train and test sets
2. The model learns patient-specific features (camera artifacts, unique anatomy) instead of disease features
3. Test performance is artificially inflated

**The Voets et al. (2019) Warning**:
> "Voets et al. demonstrated that reproducing Gulshan et al.'s results with public data yielded AUC drops of 4–14 points due to: (1) private curation steps, (2) **per-image rather than per-patient splits**, (3) label adjudication differences."

### 6.2. APTOS 2019: No Patient IDs Available

**Critical Limitation**: The APTOS 2019 dataset does not include patient identifiers. The CSV structure is:

```csv
id_code,diagnosis
000c1434d8d7,2
001639a390f0,4
```

The `id_code` is a **random hash** with no relationship to patient identity. There is no way to:
- Group images by patient
- Ensure patient-level separation in cross-validation
- Verify absence of same-patient images across splits

**Mathematical Impact on Metric Estimation**:

Let $\rho$ be the probability that a patient has multiple images in the dataset. If images are split randomly:
- Probability of patient overlap between train/val: $\rho \cdot \frac{|V|}{N}$ per patient
- Expected metric inflation: $\Delta_{\text{AUC}} \approx 0.04 - 0.14$ (Voets 2019)

### 6.3. Current Implementation: Per-Image Stratified Split

**Code Reference** (`dataset.py:332–348`):
```python
def get_train_val_split(
    df: pd.DataFrame,
    val_fold: int = 0,
    n_folds: int = N_FOLDS,  # 5
    seed: int = RANDOM_SEED,  # 42
):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["diagnosis"])):
        df.loc[val_idx, "fold"] = fold_idx

    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]
    return train_df, val_df
```

**What This Does**:
- Stratified by diagnosis class (preserves label distribution)
- Split by image index (NOT by patient)
- Deterministic via fixed seed (reproducible)

**What This Does NOT Do**:
- Group images by patient (impossible without patient IDs)
- Prevent potential patient overlap (cannot verify)
- Guarantee unbiased generalization estimates

### 6.4. Mitigation Strategies

| Strategy | Implemented | Effectiveness |
|----------|-------------|---------------|
| **External Validation on Messidor-2** | ✅ Yes | HIGH — Completely independent dataset |
| **Document as Known Limitation** | ✅ Yes | N/A — Transparency |
| **Patient ID Heuristics** (e.g., filename patterns) | ❌ No | LOW — APTOS filenames are random hashes |
| **Conservative Metric Interpretation** | ✅ Yes | MEDIUM — Report with uncertainty bounds |
| **Ensemble Across Folds** | ⚠️ Partial (fold 0 only) | MEDIUM — Reduces single-split variance |

### 6.5. External Validation: The Definitive Test

**Messidor-2 as Ground Truth**:
- Completely different patient population (French vs. Indian hospitals)
- Different cameras and imaging conditions
- Independent labeling (3 US retina specialists, adjudicated)
- Never used for training → Zero leakage risk

**Observed Performance Gap** (Phase 3D results):

| Metric | APTOS Val (Fold 0) | Messidor-2 (External) | Gap |
|--------|--------------------|-----------------------|-----|
| Accuracy | 84.45% | 63.99% | -20.5 pp |
| QWK | 0.9088 | 0.6000 | -0.31 |
| AUC (Referable) | 0.9846 | 0.8911 | -0.09 |

**Interpretation**:
- The ~20-point accuracy drop is consistent with domain shift AND potential per-image split inflation
- Messidor-2 QWK 0.60 is a more realistic estimate of real-world performance
- External validation successfully detects generalization failures

---

## 7. Training and Evaluation Pipeline Verification

### 7.1. Training Pipeline Correctness

**Question**: Are training and evaluation correctly done on the train/test sets?

**Answer**: **YES**, with the following structure:

| Dataset | Role | Split Method | Used In |
|---------|------|--------------|---------|
| APTOS train (3,662) | Training + Validation | 5-fold Stratified CV | `train.py` |
| APTOS test (1,928) | **UNUSED** | N/A (no labels) | — |
| Messidor-2 (1,745 gradable) | External Test | Held-out (100%) | `evaluate.py` |

### 7.2. Code Verification: Training

From `train.py:521–536`:
```python
# ---- Data ----
df = pd.read_csv(APTOS_TRAIN_CSV)  # Only train.csv
train_df, val_df = get_train_val_split(df, val_fold=args.fold)

# Uses APTOS_TRAIN_IMAGES (train_images/ directory)
train_loader, val_loader = create_dataloaders(
    train_df, val_df, APTOS_TRAIN_IMAGES,
    batch_size=args.batch_size, num_workers=NUM_WORKERS,
    use_cache=use_cache, cache_dir=cache_dir,
)
```

**Verification**:
- ✅ Reads `APTOS_TRAIN_CSV` (not test.csv)
- ✅ Uses `APTOS_TRAIN_IMAGES` (not test_images/)
- ✅ Applies stratified fold splitting
- ✅ Creates separate train/val loaders

### 7.3. Code Verification: Evaluation

From `evaluate.py:501–522`:
```python
# ---- Dataset ----
dataset = MessidorDataset(
    csv_path=str(MESSIDOR_CSV),        # Messidor-2 labels (Krause et al. 2018)
    image_dir=str(MESSIDOR_IMAGES),    # Messidor-2 images
    transform=get_val_transform(IMAGE_SIZE),
    preprocess=True,
    labels_available=not args.no_labels,
)
```

**Verification**:
- ✅ Uses Messidor-2 (completely separate from APTOS)
- ✅ Uses adjudicated labels from Krause et al. 2018
- ✅ Applies validation transform (no augmentation)
- ✅ Filters to gradable images only (1,745 of 1,748)

### 7.4. Data Flow Diagram (Detailed)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PHASE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  APTOS train.csv + train_images/                                        │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │               Stratified 5-Fold Split (seed=42)                  │    │
│  │                                                                   │    │
│  │   Fold 0:  Train = 2929 images  │  Val = 733 images              │    │
│  │   Fold 1:  Train = 2929 images  │  Val = 733 images              │    │
│  │   ...                                                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                    │                         │
│           ▼                                    ▼                         │
│  ┌─────────────────────┐          ┌─────────────────────┐              │
│  │   Training Set       │          │   Validation Set    │              │
│  │ (augmentation ON)    │          │ (augmentation OFF)  │              │
│  │ (Ben Graham cached)  │          │ (Ben Graham cached) │              │
│  └─────────────────────┘          └─────────────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PHASE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Messidor-2 (messidor_data.csv + IMAGES/)                               │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Filter: gradable == 1                         │    │
│  │                    (1,748 → 1,745 images)                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     External Test Set                            │    │
│  │                (augmentation OFF, T=20 MC passes)                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Identified Issues and Recommendations

### 8.1. Issue 1: Per-Image Split Inflates APTOS Metrics

**Severity**: MEDIUM
**Status**: ⚠️ UNFIXABLE (structural limitation)

**Problem**: Without patient IDs, we cannot verify that the same patient's images don't appear in both train and validation sets. This potentially inflates validation metrics by 4-14 AUC points (Voets 2019).

**Evidence**:
- APTOS Val QWK: 0.9088 (suspiciously high)
- Messidor-2 QWK: 0.6000 (more realistic)
- Gap larger than typical domain shift alone would explain

**Recommendations**:
1. **Report with caveat**: "Per-image stratified split; potential metric inflation due to missing patient IDs"
2. **Emphasize external validation**: Messidor-2 results are the reliable benchmark
3. **Compare to literature**: Voets 2019 showed similar gaps; this is a known issue
4. **Conservative interpretation**: Treat APTOS validation as upper bound

### 8.2. Issue 2: Preprocessing Versus Augmentation Confusion

**Severity**: LOW
**Status**: ✅ RESOLVED (documentation issue)

**Problem**: Some documentation mentions "CLAHE preprocessing" when CLAHE is actually used only for training augmentation, not offline preprocessing.

**Clarification**:
| Step | Timing | Location | Deterministic? |
|------|--------|----------|----------------|
| Ben Graham preprocessing | Offline (once) | `preprocess_data.py` | ✅ Yes |
| CLAHE augmentation | Runtime (training) | `dataset.py` | ❌ No (p=0.3) |

**Recommendation**: Update documentation to clearly distinguish offline preprocessing from runtime augmentation.

### 8.3. Issue 3: Preprocessed Cache Not Verified at Load Time

**Severity**: LOW
**Status**: ⚠️ NOT FIXED

**Problem**: The `DRDataset` class loads preprocessed images from cache without verifying:
- Image integrity (corrupted files)
- Preprocessing version (if algorithm changes, cache is stale)
- File count consistency (missing images)

**Code Reference** (`dataset.py:94–96`):
```python
if self.use_cache and cached_path is not None and cached_path.exists():
    image = cv2.imread(str(cached_path))
    if image is None:
        raise ValueError(f"Failed to load cached image: {cached_path}")
```

**Recommendations**:
1. Add preprocessing report hash to cache directory
2. Verify image dimensions match expected `IMAGE_SIZE`
3. Warn if cache was created with different preprocessing parameters

### 8.4. Issue 4: No Explicit Train/Test Leakage Check

**Severity**: LOW (mitigated by data structure)
**Status**: ✅ N/A (not applicable)

**Problem**: There's no automated check preventing APTOS test images from being used.

**Mitigation**: This is inherently prevented by:
- `APTOS_TRAIN_CSV` constant points only to `train.csv`
- `APTOS_TRAIN_IMAGES` constant points only to `train_images/`
- Test images never preprocessed (not in cache)

### 8.5. Issue 5: GAN Augmentation and Super-Resolution Not Implemented

**Severity**: LOW (scope limitation)
**Status**: ⬜ FUTURE WORK

**Problem**: The checklist mentions "GAN augmentation, super-resolution" but these are not implemented.

**Justification**:
- Bachelor's thesis scope constraints
- Standard augmentations sufficient for proof-of-concept
- Complex implementation with uncertain benefit

**Recommendation**: Document as future work, not as missing functionality.

---

## 9. Mathematical Analysis of Preprocessing Impact

### 9.1. Ben Graham's Local Color Normalization: Frequency Domain Analysis

The preprocessing formula:
$$
y(x) = 4 \cdot f(x) - 4 \cdot (f * G_\sigma)(x) + 128
$$

Where $G_\sigma$ is a Gaussian kernel, can be rewritten as:
$$
y(x) = 4 \cdot f(x) \cdot (1 - G_\sigma(x)) + 128
$$

In the frequency domain (via convolution theorem):
$$
Y(\omega) = 4 \cdot F(\omega) \cdot (1 - \hat{G}_\sigma(\omega))
$$

Where $\hat{G}_\sigma(\omega) = e^{-\frac{\omega^2 \sigma^2}{2}}$ is the Fourier transform of the Gaussian.

**Key Insight**: $1 - \hat{G}_\sigma(\omega)$ is a high-pass filter that:
- Suppresses $\omega \approx 0$ (DC component = average illumination)
- Passes $\omega > \frac{1}{\sigma}$ (fine details like lesions)

For $\sigma = \frac{512}{30} \approx 17.07$ pixels:
- Cutoff frequency: $f_c \approx \frac{1}{17.07} \approx 0.06$ cycles/pixel
- Features smaller than ~17 pixels (e.g., microaneurysms ~10-15px) are preserved
- Features larger than ~100 pixels (e.g., illumination gradients) are suppressed

### 9.2. Circular Masking: Information Loss Analysis

When a circular mask is applied to a square image, the masked area fraction is:
$$
\text{Preserved fraction} = \frac{\pi r^2}{(2r)^2} = \frac{\pi}{4} \approx 78.5\%
$$

This means 21.5% of pixels are zeroed. However, these are camera surround pixels containing:
- Vignetting artifacts
- Timestamp/text overlays
- Black padding

**Net effect**: Information loss is negligible for clinical features; noise reduction is significant.

### 9.3. Stratified Split: Expected Class Distribution

For 5-fold CV with stratification, each fold approximately preserves:

| Grade | Full Dataset | Expected per Fold (20%) |
|-------|--------------|-------------------------|
| 0     | 1,805 (49.3%) | ~361 (49.3%) |
| 1     | 370 (10.1%) | ~74 (10.1%) |
| 2     | 999 (27.3%) | ~200 (27.3%) |
| 3     | 193 (5.3%) | ~39 (5.3%) |
| 4     | 295 (8.1%) | ~59 (8.1%) |

**Variance bound**: With stratification, the maximum deviation from expected proportion per class is bounded by:
$$
\epsilon \leq \frac{1}{\sqrt{n_{\text{class}}}} \cdot z_{\alpha/2}
$$

For the smallest class (Grade 3, n=193), a 20% validation fold has ~39 samples with standard error $\frac{1}{\sqrt{39}} \approx 0.16$, giving 95% CI of ~$\pm 30\%$ relative deviation.

---

## 10. Summary of Findings

### 10.1. Answers to Original Questions

**Q1: Is training/evaluation correctly done on train/test sets?**

**A:** YES, the pipeline correctly:
- Uses only APTOS training data (3,662 labeled images) for training/validation
- Applies 5-fold stratified cross-validation
- Uses Messidor-2 (1,745 images) exclusively for external testing
- Never uses APTOS test set (no labels available)

**Q2: Why doesn't `data/processed/aptos` have train/test subfolders?**

**A:** By design, because:
- APTOS test set has no ground-truth labels (Kaggle competition format)
- Processing it would serve no purpose
- Train/val splits are computed dynamically at runtime via 5-fold CV
- Messidor-2 serves as the external test set

**Q3: Are preprocessing steps documented with patient-level splits verified?**

**A:** Partially:
- ✅ Preprocessing steps are well-documented (Ben Graham's circular crop + local color normalization)
- ⚠️ CLAHE is used in augmentation, not preprocessing (documentation clarified)
- ❌ Patient-level splits are impossible to verify (APTOS lacks patient IDs)
- ❌ GAN augmentation and super-resolution are not implemented (scope limitation)

### 10.2. Leakage Risk Assessment

| Leakage Type | Risk Level | Mitigation |
|--------------|------------|------------|
| **Train → APTOS Test** | NONE | Test set never used (no labels) |
| **Train → Messidor-2** | NONE | Completely separate dataset |
| **Same-patient train/val** | UNKNOWN | APTOS lacks patient IDs |
| **Preprocessing information** | NONE | Deterministic, label-agnostic |
| **Augmentation information** | NONE | Applied only at runtime |

### 10.3. Recommendations Prioritized

| Priority | Recommendation | Effort | Impact |
|----------|----------------|--------|--------|
| HIGH | Report per-image split limitation in thesis | Low | Transparency |
| HIGH | Emphasize Messidor-2 as primary benchmark | Low | Valid conclusions |
| MEDIUM | Implement preprocessing cache validation | Medium | Robustness |
| LOW | Update documentation re: CLAHE vs preprocessing | Low | Clarity |
| FUTURE | Consider GAN augmentation for minority classes | High | Uncertain |

---

## Appendix A: Quick Reference Commands

### Preprocess APTOS data:
```bash
python src/preprocess_data.py --dataset aptos --workers 4 --visualize
```

### Verify processed count:
```bash
ls data/processed/aptos/ | wc -l  # Should output 3662
```

### Check class distribution:
```python
import pandas as pd
df = pd.read_csv("aptos/aptos2019-blindness-detection/train.csv")
print(df["diagnosis"].value_counts().sort_index())
```

### Run training (fold 0):
```bash
python src/train.py --model cbam --fold 0 --epochs 20 --use_cache
```

### Run evaluation on Messidor-2:
```bash
python src/evaluate.py --checkpoint "outputs/checkpoints/cbam_resnet50*fold0_best.pth" --mc_passes 20
```

---

## Appendix B: File Hash Verification (Data Integrity)

For reproducibility, the following file counts should be verified:

| Path | Expected Count | Command |
|------|----------------|---------|
| `aptos/.../train.csv` | 3,663 lines | `wc -l train.csv` |
| `aptos/.../train_images/` | 3,662 files | `ls train_images/ \| wc -l` |
| `aptos/.../test.csv` | 1,929 lines | `wc -l test.csv` |
| `aptos/.../test_images/` | 1,928 files | `ls test_images/ \| wc -l` |
| `data/processed/aptos/` | 3,662 files | `ls data/processed/aptos/ \| wc -l` |

---

_This document provides a comprehensive analysis of the data preprocessing pipeline. For implementation changes, consult `src/preprocessing.py`, `src/preprocess_data.py`, and `src/dataset.py`._
