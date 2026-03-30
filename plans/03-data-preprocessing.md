# Data Preprocessing Implementation Plan

> **Document Version**: 1.2
> **Created**: 2026-03-30
> **Updated**: 2026-03-30
> **Scope**: Offline preprocessing pipeline for APTOS and Messidor-2 datasets
> **Status**: ✅ Phases 1-3 COMPLETED | ⚠️ Phase 4 (Integration) NOT IMPLEMENTED

---

## Executive Summary

This plan implements offline preprocessing for both datasets. The goal is to preprocess all images **once** using Ben Graham's pipeline and save them to `data/processed/`, eliminating redundant computation during training.

**Output directories**:
- `data/processed/aptos/` — Preprocessed APTOS 2019 images (3,662 images) ✅ COMPLETED
- `data/processed/messidor2/` — Preprocessed Messidor-2 images (1,058 images) ✅ COMPLETED
  - *Note: Actual dataset contains 1,058 images, not 690 as originally estimated*

**Key deliverables**:
1. ✅ `src/preprocess_data.py` — Offline preprocessing script with parallel processing
2. ✅ Preprocessed images saved as PNG files
3. ✅ Preprocessing statistics report (JSON) — `data/processed/preprocessing_report.json`
4. ✅ Optional before/after visualization samples

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Implementation Plan](#2-implementation-plan)
3. [Technical Specification](#3-technical-specification)
4. [TODO Checklist](#4-todo-checklist)
5. [Testing & Verification](#5-testing--verification)

---

## 1. Current State Analysis

### 1.1 Existing Preprocessing Implementation

**File: `src/preprocessing.py`**

The current implementation follows Ben Graham's Kaggle-winning approach:

```python
def ben_graham_preprocess(image: np.ndarray, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """Complete Ben Graham preprocessing pipeline.

    1. Circular crop — removes black borders, resizes to target_size.
    2. Local color normalization — subtracts Gaussian-blurred local mean.
    """
    cropped = circular_crop(image, target_size)
    result = local_color_normalization(cropped)
    return result
```

**Step 1 — Circular Crop (`circular_crop`)**:
- Detects retina boundary via grayscale thresholding + contour detection
- Crops to bounding box with 5% margin
- Resizes to `IMAGE_SIZE` (512×512)
- Applies circular mask to zero-out pixels outside retina disc

**Step 2 — Local Color Normalization (`local_color_normalization`)**:
- Formula: `result = 4 * image - 4 * GaussianBlur(image, σ) + 128`
- σ = width / 30 (adaptive to image resolution)
- Removes illumination variation while preserving lesion detail

### 1.2 Current Dataset Integration

**File: `src/dataset.py`**

Preprocessing is applied on-the-fly in `__getitem__`:

```python
class DRDataset(Dataset):
    def __getitem__(self, idx):
        image = cv2.imread(str(img_path))

        if self.preprocess:
            image = ben_graham_preprocess(image, self.target_size)  # <- ON EVERY LOAD

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ... augmentation, normalization, return
```

**Training Augmentation Pipeline** (`get_train_transform`):
```python
A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift=0.1, scale=0.1, rotate=45),
    A.OneOf([RandomBrightnessContrast, CLAHE], p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.2),  # <- DEPRECATED API
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
```

**Validation Transform** (`get_val_transform`):
```python
A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
```

### 1.3 Evaluation State

**File: `src/evaluate.py`**

Currently only evaluates on Messidor-2:
- Loads `MessidorDataset` with preprocessing
- Runs MC Dropout inference
- Computes metrics + uncertainty analysis

**Missing**: APTOS validation set evaluation with same MC Dropout uncertainty analysis.

---

## 2. Implementation Plan

### 2.1 Performance Impact

**Current** (on-the-fly preprocessing):
- APTOS train: ~2,930 images × 20 epochs = 58,600 preprocessing operations
- Ben Graham preprocessing time: ~50-100ms per image
- Total overhead: ~50-100 minutes added to training

**With offline preprocessing**:
- Preprocess once: ~5 minutes (with 4 parallel workers)
- Training reads cached images: near-zero overhead
- ~30% faster training

### 2.2 Output Structure

```
data/
└── processed/
    ├── aptos/
    │   ├── 000c1434d8d7.png      # Preprocessed APTOS images
    │   ├── 001639a390f0.png
    │   └── ... (3,662 files)
    ├── messidor2/
    │   ├── 20051020_43808_0100_PP.png  # Preprocessed Messidor-2 images
    │   └── ... (690 files)
    └── preprocessing_report.json       # Statistics summary
```

### 2.3 Script Interface

```bash
# Process both datasets
python src/preprocess_data.py

# Process APTOS only
python src/preprocess_data.py --dataset aptos

# Process Messidor-2 only
python src/preprocess_data.py --dataset messidor

# With visualization samples
python src/preprocess_data.py --visualize

# Custom settings
python src/preprocess_data.py --workers 8 --image_size 512
```

---

## 3. Technical Specification

### 3.1 Core Functions

**`preprocess_single_image()`**
- Loads image via cv2.imread
- Applies Ben Graham pipeline (circular crop + color normalization)
- Saves preprocessed image as PNG
- Returns: (success, radius, used_fallback, error_message)

**`preprocess_dataset()`**
- Iterates over DataFrame with image IDs
- Uses `ProcessPoolExecutor` for parallel processing
- Collects preprocessing statistics
- Saves JSON error log if any failures

**`find_retina_circle()` modification**
- Returns additional `used_fallback: bool` flag
- Logs warning when contour detection fails

### 3.2 Preprocessing Statistics

The script should collect and report:

| Statistic | Description |
|-----------|-------------|
| `total_images` | Number of images in dataset |
| `successful` | Successfully preprocessed count |
| `failed` | Failed to preprocess count |
| `fallback_used` | Images where contour detection failed |
| `mean_radius` | Average detected retina radius (px) |
| `min_radius` | Minimum detected radius |
| `max_radius` | Maximum detected radius |

### 3.3 Dataset Integration (Future)

After preprocessing, update `dataset.py` to support cached images:

```python
class DRDataset(Dataset):
    def __init__(
        self,
        ...,
        use_cache: bool = False,
        cache_dir: str | None = None,
    ):
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __getitem__(self, idx):
        if self.use_cache and self.cache_dir:
            # Load preprocessed image directly
            img_path = self.cache_dir / f"{image_id}.png"
            image = cv2.imread(str(img_path))
        else:
            # Original on-the-fly preprocessing
            ...
```

---

## 4. TODO Checklist

### Phase 1: Core Script Implementation ✅ COMPLETED

- [x] **1.1** Create `src/preprocess_data.py` skeleton
  - [x] Add argparse CLI interface
  - [x] Import config paths and preprocessing functions
  - [x] Set up logging

- [x] **1.2** Implement `preprocess_single_image()` function
  - [x] Load image via cv2
  - [x] Apply `ben_graham_preprocess()`
  - [x] Save to output path as PNG
  - [x] Return success status and statistics

- [x] **1.3** Implement `preprocess_dataset()` function
  - [x] Accept DataFrame, source_dir, output_dir parameters
  - [x] Build task list from DataFrame rows
  - [x] Use `ProcessPoolExecutor` for parallel processing
  - [x] Show progress with tqdm
  - [x] Collect and aggregate statistics

- [x] **1.4** Implement main entry point
  - [x] Parse CLI arguments (--dataset, --workers, --visualize)
  - [x] Call `seed_everything()` and `setup_directories()`
  - [x] Process APTOS if selected
  - [x] Process Messidor-2 if selected
  - [x] Save preprocessing report JSON

### Phase 2: Preprocessing Statistics ✅ COMPLETED

- [x] **2.1** Modify `find_retina_circle()` in `preprocessing.py`
  - [x] *Implementation Note:* Fallback detection implemented in `preprocess_data.py` (lines 119-123) instead of modifying `find_retina_circle()`. Works by comparing detected radius to expected fallback radius.

- [x] **2.2** Create `PreprocessingStats` dataclass
  - [x] Fields: total_images, successful, failed, fallback_used
  - [x] Fields: mean_radius, min_radius, max_radius
  - [x] Method to serialize to dict (via `asdict()`)

- [x] **2.3** Save preprocessing report
  - [x] Save to `data/processed/preprocessing_report.json`
  - [x] Include timestamp, image_size, per-dataset stats
  - [x] Save error log if any failures

### Phase 3: Optional Visualization ✅ COMPLETED

- [x] **3.1** Implement `save_visualization_samples()` function
  - [x] Sample images from each DR grade (if available)
  - [x] Create side-by-side before/after grid
  - [x] Save to `outputs/figures/preprocessing_*.png`

### Phase 4: Integration ⚠️ NOT IMPLEMENTED (Deferred)

- [ ] **4.1** Update config.py
  - [ ] Add `APTOS_PROCESSED_DIR = PROCESSED_DIR / "aptos"`
  - [ ] Add `MESSIDOR_PROCESSED_DIR = PROCESSED_DIR / "messidor2"`
  - [ ] Add `USE_PREPROCESSED_CACHE = True` flag

- [ ] **4.2** Update `DRDataset` class
  - [ ] Add `use_cache: bool` parameter
  - [ ] Add `cache_dir: str | None` parameter
  - [ ] Load from cache when enabled

- [ ] **4.3** Update `MessidorDataset` class
  - [ ] Add `use_cache: bool` parameter
  - [ ] Load from cache when enabled

- [ ] **4.4** Update `train.py`
  - [ ] Add `--use_cache` CLI argument
  - [ ] Pass cache directory to dataloaders

**⚠️ IMPORTANT:** Until Phase 4 is implemented, the training pipeline will continue to apply Ben Graham preprocessing on-the-fly during training, despite preprocessed images being available in `data/processed/`.

---

## 5. Testing & Verification

### 5.1 Smoke Test

After implementation, run:

```bash
# Process both datasets
python src/preprocess_data.py --visualize

# Verify output
ls data/processed/aptos/ | wc -l      # Should be 3662
ls data/processed/messidor2/ | wc -l  # Should be 690

# Check report
cat data/processed/preprocessing_report.json
```

### 5.2 Actual Report ✅

```json
{
  "timestamp": "2026-03-30T23:35:53.195303",
  "image_size": 512,
  "datasets": {
    "aptos": {
      "total_images": 3662,
      "successful": 3662,
      "failed": 0,
      "fallback_used": 0,
      "mean_radius": 939.413981430912,
      "min_radius": 247,
      "max_radius": 1856,
      "radii": null
    },
    "messidor2": {
      "total_images": 1058,
      "successful": 1058,
      "failed": 0,
      "fallback_used": 0,
      "mean_radius": 591.8317580340265,
      "min_radius": 464,
      "max_radius": 747,
      "radii": null
    }
  }
}
```

**Key Observations:**
- ✅ All images preprocessed successfully (0 failures)
- ✅ No fallback used (all images had successful contour detection)
- ℹ️ Messidor-2 contains 1,058 images (not 690 as estimated)
- ℹ️ Mean radius values are larger than expected, indicating high-resolution source images

### 5.3 Verification Checklist ✅ ALL VERIFIED

- [x] All APTOS images preprocessed (3,662 files in output)
- [x] All Messidor-2 images preprocessed (1,058 files in output)
- [x] No failed images in report
- [x] Visualization samples look correct (circular crop + color normalization applied)
- [x] Preprocessing report saved with valid statistics

---

## File Changes Summary

| File | Change Type | Status | Description |
|------|-------------|--------|-------------|
| `src/preprocess_data.py` | **New** | ✅ DONE | Offline preprocessing script with parallel processing |
| `src/preprocessing.py` | Modify | ⚠️ PARTIAL | Fallback tracking implemented in `preprocess_data.py` instead |
| `src/config.py` | Modify | ❌ TODO | Add cache paths (Phase 4 - deferred) |
| `src/dataset.py` | Modify | ❌ TODO | Add cache support (Phase 4 - deferred) |
| `src/train.py` | Modify | ❌ TODO | Add `--use_cache` argument (Phase 4 - deferred) |

---

## Current Usage Status

**Preprocessing script is functional and has been executed:**
```bash
# Already run successfully on 2026-03-30
python src/preprocess_data.py --visualize
```

**To use preprocessed images in training (Phase 4 required):**
```bash
# NOT YET IMPLEMENTED - requires Phase 4 integration
python src/train.py --use_cache
```

**Current training behavior:**
- Training still uses on-the-fly preprocessing via `ben_graham_preprocess()`
- Preprocessed images exist in `data/processed/` but are not used
- To enable cached preprocessing, implement Phase 4 integration tasks

---

*Document updated: 2026-03-30 (v1.2)*
