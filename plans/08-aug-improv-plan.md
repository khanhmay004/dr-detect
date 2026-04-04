# Plan 08 — Augmentation-Based Class Balancing: Implementation

> **Spec doc**: `docs/08-aug-improv.md`
> **Status**: Ready to implement
> **Date**: 2026-04-04
> **Builds on**: Phase 7B post-mortem; original baseline (20260403_120248) retained as anchor

---

## Overview

This plan converts the design in `docs/08-aug-improv.md` into concrete, file-level code changes. The work is organised into four phases:

| Phase | Scope | Retraining? | Risk |
|-------|-------|-------------|------|
| **Phase 0** | Infrastructure — config, dataset, config system | No | None |
| **Phase 1** | Training integration — train.py wiring + YAML | No | None |
| **Phase 2** | Validation — CPU smoke test + unit checks | No | None |
| **Phase 3** | GPU experiments — Run B (quantity) → Run C (full) + Messidor-2 eval | Yes | Medium |

**Cardinal rule from Phase 7B**: If Messidor-2 Grade 0 recall drops below 85%, the experiment is failing. Roll back and lower `aug_target_count_per_class`.

---

## Phase 0 — Infrastructure

### Task 0.1 — `config.py`: Add new constants

**File**: `src/config.py`
**Where**: After the existing `IMPROVEMENT PARAMETERS (Plan 07)` block.

```python
# --- Augmentation-based balancing (Plan 08) ---
USE_AUG_BALANCED_DATASET: bool = False     # False = existing DRDataset (default, safe)
AUG_TARGET_COUNT_PER_CLASS: int = 800      # Minimum samples per class after expansion
AUG_FOCAL_ALPHA_UNIFORM: bool = True       # True = alpha=[1,1,1,1,1] when aug balanced

# Grade → augmentation intensity level.
# None = use DEFAULT_GRADE_TO_AUG_LEVEL in dataset.py (recommended).
# Override format: {0: "standard", 1: "heavy", ...}
AUG_GRADE_LEVEL_OVERRIDE: dict[int, str] | None = None
```

No other changes to `config.py`.

---

### Task 0.2 — `dataset.py`: Add type alias and grade-to-level map

**File**: `src/dataset.py`
**Where**: After the existing `import` block, before `class DRDataset`.

Add the `AugLevel` type alias and the default grade-to-augmentation-level mapping:

```python
from typing import Literal

# Augmentation intensity level for class-conditional transforms (Plan 08).
AugLevel = Literal["standard", "medium", "heavy"]

# Default mapping: grade → augmentation intensity, keyed on Messidor-2 recall.
# Grade 0: 94.2% recall — already well-classified, no intervention.
# Grade 1:  9.3% recall — worst performer, maximum domain-variation transforms.
# Grade 2: 16.1% recall — referable class, clinically critical failure.
# Grade 3: 56.0% recall — moderate failure, medium intervention.
# Grade 4: 40.0% recall — significant failure, medium-heavy intervention.
DEFAULT_GRADE_TO_AUG_LEVEL: dict[int, AugLevel] = {
    0: "standard",
    1: "heavy",
    2: "heavy",
    3: "medium",
    4: "medium",
}
```

---

### Task 0.3 — `dataset.py`: Add three transform factory functions

**File**: `src/dataset.py`
**Where**: After the existing `get_val_transform()` function (after line ~300).

These three functions replace the single `get_train_transform()` with a tiered system. All three remain compatible with the existing pipeline — they produce the same `A.Compose` output type.

```python
# =========================================================================
#  Class-conditional augmentation transform factories (Plan 08)
#  Three tiers: standard (Grade 0) → medium (Grades 3,4) → heavy (Grades 1,2)
# =========================================================================

def _build_standard_transform(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Standard augmentation pipeline — identical to existing get_train_transform().

    Applied to Grade 0 (majority class, 94.2% Messidor-2 recall). No
    additional intervention needed; this is the baseline augmentation.

    Args:
        image_size: Spatial resolution after resize.

    Returns:
        Albumentations Compose pipeline.

    Example:
        >>> t = _build_standard_transform(256)
        >>> out = t(image=np.zeros((256, 256, 3), dtype=np.uint8))
        >>> out["image"].shape
        torch.Size([3, 256, 256])
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=1.0
            ),
            A.CLAHE(clip_limit=2.0, p=1.0),
        ], p=0.3),
        A.GaussNoise(std_range=(0.04, 0.20), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _build_medium_transform(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Medium augmentation — for Grades 3 and 4 (moderate failure classes).

    Adds colour-domain transforms (HueSaturationValue, RandomGamma,
    ImageCompression) on top of the standard pipeline. Designed to simulate
    the camera and domain variation between APTOS and Messidor-2 without
    being so aggressive that fine lesion structure is destroyed.

    Args:
        image_size: Spatial resolution after resize.

    Returns:
        Albumentations Compose pipeline.

    Example:
        >>> t = _build_medium_transform(256)
        >>> out = t(image=np.zeros((256, 256, 3), dtype=np.uint8))
        >>> out["image"].shape
        torch.Size([3, 256, 256])
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            scale=(0.85, 1.15),
            rotate=(-60, 60),
            mode=cv2.BORDER_CONSTANT,
            p=0.6,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=1.0
            ),
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        ], p=0.5),
        # Simulates different fundus camera colour calibrations
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4,
        ),
        A.GaussNoise(std_range=(0.04, 0.25), p=0.3),
        # JPEG compression artifacts common in clinical imaging archives
        A.ImageCompression(quality_range=(75, 100), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _build_heavy_transform(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Heavy augmentation — for Grades 1 and 2 (worst-performing classes).

    The most aggressive pipeline, targeting the two classes that fail
    catastrophically on Messidor-2 (Grade 1: 9.3%, Grade 2: 16.1% recall).
    Adds elastic distortion, CoarseDropout, MotionBlur, and stronger colour
    manipulation.

    Design intent: Each augmented copy should look like it came from a
    different fundus camera under different conditions — building cross-domain
    invariance for microaneurysm and exudate detection.

    Args:
        image_size: Spatial resolution after resize.

    Returns:
        Albumentations Compose pipeline.

    Example:
        >>> t = _build_heavy_transform(256)
        >>> out = t(image=np.zeros((256, 256, 3), dtype=np.uint8))
        >>> out["image"].shape
        torch.Size([3, 256, 256])
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            scale=(0.85, 1.15),
            rotate=(-90, 90),   # Full range — retinal images are rotation-invariant
            mode=cv2.BORDER_CONSTANT,
            p=0.7,
        ),
        # Elastic/grid distortions simulate retinal curvature and lens distortion
        # differences across acquisition systems
        A.OneOf([
            A.ElasticTransform(alpha=1.0, sigma=50.0, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.4, contrast_limit=0.4, p=1.0
            ),
            A.CLAHE(clip_limit=6.0, p=1.0),
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),
        ], p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.5,
        ),
        # CoarseDropout simulates vitreous floaters, drusen, and imaging artifacts
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 48),
            hole_width_range=(16, 48),
            fill_value=0,
            p=0.3,
        ),
        # MotionBlur simulates patient eye movement during acquisition
        A.MotionBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(std_range=(0.04, 0.30), p=0.4),
        A.ImageCompression(quality_range=(65, 95), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
```

---

### Task 0.4 — `dataset.py`: Add `AugmentedBalancedDataset` class

**File**: `src/dataset.py`
**Where**: After the three transform factory functions above.

```python
# =========================================================================
#  AugmentedBalancedDataset (Plan 08)
# =========================================================================

class AugmentedBalancedDataset(Dataset):
    """Training dataset that oversamples minority classes via augmentation.

    Builds an expanded sample index so each class reaches ``target_count_per_class``
    samples. Minority class samples appear multiple times, but each repetition
    receives a freshly sampled stochastic augmentation — so every "copy" is
    visually distinct. This is fundamentally different from WeightedRandomSampler,
    which repeats exact pixels (see docs/08-aug-improv.md §2 for the distinction).

    Class-conditional augmentation intensity is controlled via
    ``grade_to_aug_level``: poorly-classified grades receive heavier transforms
    to build domain-invariant representations.

    Args:
        df: Training DataFrame with ``id_code`` and ``diagnosis`` columns.
        image_dir: Directory containing ``.png`` fundus images.
        target_count_per_class: Minimum samples per class after expansion.
            Classes already at or above this count are NOT downsampled.
        grade_to_aug_level: Mapping from DR grade (0–4) to augmentation
            intensity. Defaults to DEFAULT_GRADE_TO_AUG_LEVEL.
        image_size: Target spatial resolution.
        preprocess: Apply Ben Graham crop + Gaussian-blur normalisation.
        use_cache: Load from preprocessed cache.
        cache_dir: Path to cache directory (None = no cache).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "id_code": [f"img_{i}" for i in range(10)],
        ...     "diagnosis": [0, 0, 0, 0, 0, 1, 1, 2, 3, 4],
        ... })
        >>> ds = AugmentedBalancedDataset(
        ...     df=df, image_dir=".", target_count_per_class=5,
        ...     preprocess=False,
        ... )
        >>> # Grades 1,3,4 each had <5 samples → expanded to 5
        >>> len(ds) >= 5 * 5
        True
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        target_count_per_class: int = AUG_TARGET_COUNT_PER_CLASS,
        grade_to_aug_level: dict[int, AugLevel] | None = None,
        image_size: int = IMAGE_SIZE,
        preprocess: bool = True,
        use_cache: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.grade_to_aug_level = grade_to_aug_level or DEFAULT_GRADE_TO_AUG_LEVEL
        self.image_size = image_size
        self.preprocess = preprocess
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        # Build per-level transform pipelines once at init (not per-sample)
        self._transforms: dict[AugLevel, A.Compose] = {
            "standard": _build_standard_transform(image_size),
            "medium":   _build_medium_transform(image_size),
            "heavy":    _build_heavy_transform(image_size),
        }

        # Build the expanded (df_row_index, grade) sample index
        self._index: list[tuple[int, int]] = self._build_index(
            target_count_per_class
        )

        # Expose .df for downstream alpha weight computation in train.py
        # (mirrors the DRDataset interface)

        # Log expansion statistics
        from collections import Counter
        expanded_counts = Counter(grade for _, grade in self._index)
        original_counts = df["diagnosis"].value_counts().sort_index().to_dict()
        logging.info(
            "AugmentedBalancedDataset: index built (target=%d). "
            "Per-class original→expanded: %s",
            target_count_per_class,
            {
                g: f"{original_counts.get(g, 0)}→{expanded_counts.get(g, 0)}"
                for g in range(NUM_CLASSES)
            },
        )

    def _build_index(self, target_count: int) -> list[tuple[int, int]]:
        """Build the expanded (row_index, grade) list.

        For each grade:
        - All original sample row indices are included first.
        - If count < target_count, indices are cycled (with wraparound) to
          reach target_count. Each cycle triggers a different random
          augmentation at __getitem__ time — they are not pixel-identical.

        Args:
            target_count: Minimum number of samples per grade.

        Returns:
            List of (df_row_index, grade) tuples.
        """
        index: list[tuple[int, int]] = []
        for grade in range(NUM_CLASSES):
            grade_row_indices = self.df[
                self.df["diagnosis"] == grade
            ].index.tolist()

            n_source = len(grade_row_indices)
            if n_source == 0:
                logging.warning(
                    "AugmentedBalancedDataset: grade %d has 0 samples — skipped.",
                    grade,
                )
                continue

            n_target = max(n_source, target_count)

            # All original samples
            for row_idx in grade_row_indices:
                index.append((row_idx, grade))

            # Augmented copies by cycling through the source pool
            n_needed = n_target - n_source
            for i in range(n_needed):
                row_idx = grade_row_indices[i % n_source]
                index.append((row_idx, grade))

        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row_idx, grade = self._index[idx]
        row = self.df.iloc[row_idx]
        image_id = str(row["id_code"])

        # --- Load image ---
        cached_path = (
            self.cache_dir / f"{image_id}.png" if self.cache_dir else None
        )
        if self.use_cache and cached_path is not None and cached_path.exists():
            image = cv2.imread(str(cached_path))
            if image is None:
                logging.error("Failed to load cached image: %s", cached_path)
                raise ValueError(f"Failed to load cached image: {cached_path}")
        else:
            img_path = self.image_dir / f"{image_id}.png"
            if not img_path.exists():
                logging.error("Image not found: %s", img_path)
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Failed to load: {img_path}")
            if self.preprocess:
                image = ben_graham_preprocess(image, self.image_size)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Class-conditional augmentation ---
        # grade_to_aug_level defaults are set from Messidor-2 per-class recall.
        # Each call to this transform produces a different random output.
        aug_level: AugLevel = self.grade_to_aug_level.get(grade, "standard")
        transform = self._transforms[aug_level]
        augmented = transform(image=image)
        image_tensor: torch.Tensor = augmented["image"]  # [C, H, W]

        return image_tensor, grade
```

**Important**: Also add `AUG_TARGET_COUNT_PER_CLASS` to the import in `dataset.py`'s `from config import ...` block:

```python
# In the existing "from config import (...)" block at the top of dataset.py,
# add these four new constants:
from config import (
    ...,  # existing imports unchanged
    USE_AUG_BALANCED_DATASET,
    AUG_TARGET_COUNT_PER_CLASS,
    AUG_FOCAL_ALPHA_UNIFORM,
    AUG_GRADE_LEVEL_OVERRIDE,
)
```

Also add `import logging` at the top of `dataset.py` (it is not currently imported).

---

### Task 0.5 — `dataset.py`: Update `create_dataloaders` signature and body

**File**: `src/dataset.py`
**Where**: The existing `create_dataloaders` function (~line 366).

Two changes:
1. Add two new keyword arguments.
2. Add a branch that instantiates `AugmentedBalancedDataset` instead of `DRDataset` when the flag is set.
3. Guard against simultaneous use with `use_balanced_sampler`.

```python
def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    image_size: int = IMAGE_SIZE,
    preprocess: bool = True,
    use_cache: bool = False,
    cache_dir: Path | None = None,
    use_balanced_sampler: bool = False,
    # ---- Plan 08 additions ----
    use_aug_balanced_dataset: bool = False,
    aug_target_count_per_class: int = AUG_TARGET_COUNT_PER_CLASS,
) -> Tuple[DataLoader, DataLoader]:
    """Create paired train/val DataLoaders for APTOS.

    Args:
        ...existing args...
        use_aug_balanced_dataset: If True, use AugmentedBalancedDataset for
            training — oversamples minority classes via diverse augmentation.
            Mutually exclusive with use_balanced_sampler.
        aug_target_count_per_class: Target sample count per class after
            augmentation expansion. Ignored if use_aug_balanced_dataset=False.

    Returns:
        Tuple of (train_loader, val_loader).

    Raises:
        ValueError: If both use_aug_balanced_dataset and use_balanced_sampler
            are True — they both balance the input distribution via different
            mechanisms and must not be stacked.
    """
    if use_aug_balanced_dataset and use_balanced_sampler:
        raise ValueError(
            "use_aug_balanced_dataset and use_balanced_sampler are mutually "
            "exclusive — both attempt to balance class distribution. "
            "Use one or the other."
        )

    if use_aug_balanced_dataset:
        # Plan 08: expanded index + class-conditional transforms
        train_dataset = AugmentedBalancedDataset(
            df=train_df,
            image_dir=image_dir,
            target_count_per_class=aug_target_count_per_class,
            image_size=image_size,
            preprocess=preprocess,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
        logging.info(
            "AugmentedBalancedDataset: %d training samples "
            "(original: %d, target per class: %d)",
            len(train_dataset), len(train_df), aug_target_count_per_class,
        )
    else:
        # Standard path — existing behaviour unchanged
        train_dataset = DRDataset(
            df=train_df, image_dir=image_dir,
            transform=get_train_transform(image_size),
            preprocess=preprocess, target_size=image_size,
            use_cache=use_cache, cache_dir=cache_dir,
        )

    val_dataset = DRDataset(   # validation is NEVER augmented with the balanced pipeline
        df=val_df, image_dir=image_dir,
        transform=get_val_transform(image_size),
        preprocess=preprocess, target_size=image_size,
        use_cache=use_cache, cache_dir=cache_dir,
    )

    sampler = None
    shuffle = True
    if use_balanced_sampler:
        train_labels = train_df["diagnosis"].values.tolist()
        sampler = make_balanced_sampler(train_labels)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
```

---

## Phase 1 — Training Integration

### Task 1.1 — `experiment_config.py`: Add Plan 08 fields

**File**: `src/configs/experiment_config.py`
**Where**: The `ExperimentConfig` dataclass, after the `# Plan 07 improvement fields` block.

```python
@dataclass
class ExperimentConfig:
    # ... existing fields unchanged ...

    # Plan 07 improvement fields
    label_smoothing: float = LABEL_SMOOTHING
    use_balanced_sampler: bool = USE_BALANCED_SAMPLER
    classifier_hidden_dim: int = CLASSIFIER_HIDDEN_DIM
    classifier_dropout_rate: float = CLASSIFIER_DROPOUT_RATE
    lr_warmup_epochs: int = LR_WARMUP_EPOCHS

    # ---- Plan 08 additions ----
    use_aug_balanced_dataset: bool = False
    aug_target_count_per_class: int = 800
    aug_focal_alpha_uniform: bool = True
```

Also update `load_config()` to pass these through:

```python
def load_config(config_path: str, cli_args: Any) -> Any:
    config = ExperimentConfig.from_yaml(config_path)

    # ... existing assignments unchanged ...

    # Plan 08 additions
    cli_args.use_aug_balanced_dataset = config.use_aug_balanced_dataset
    cli_args.aug_target_count_per_class = config.aug_target_count_per_class
    cli_args.aug_focal_alpha_uniform = config.aug_focal_alpha_uniform

    return cli_args
```

Also add imports for the new config constants at the top of `experiment_config.py`:

```python
from config import (
    ...,  # existing imports
    USE_AUG_BALANCED_DATASET,
    AUG_TARGET_COUNT_PER_CLASS,
    AUG_FOCAL_ALPHA_UNIFORM,
)
```

---

### Task 1.2 — `train.py`: Add CLI arguments

**File**: `src/train.py`
**Where**: The `argparse` block inside `main()`, after the existing `--lr_warmup_epochs` argument.

```python
# Plan 08 augmentation balancing arguments
parser.add_argument(
    "--use_aug_balanced_dataset",
    action="store_true",
    default=False,
    help="Use AugmentedBalancedDataset (oversamples minority classes via augmentation)",
)
parser.add_argument(
    "--aug_target_count_per_class",
    type=int,
    default=800,
    help="Target sample count per class when --use_aug_balanced_dataset is active",
)
parser.add_argument(
    "--aug_focal_alpha_uniform",
    action="store_true",
    default=True,
    help="Use uniform alpha=[1,1,1,1,1] when aug balanced dataset is active",
)
```

---

### Task 1.3 — `train.py`: Update `create_dataloaders` call

**File**: `src/train.py`
**Where**: The `create_dataloaders(...)` call inside `main()` (~line 487).

The existing call:
```python
train_loader, val_loader = create_dataloaders(
    train_df, val_df, APTOS_TRAIN_IMAGES,
    batch_size=args.batch_size, num_workers=NUM_WORKERS,
    use_cache=use_cache, cache_dir=cache_dir,
    use_balanced_sampler=args.use_balanced_sampler,
)
```

Updated call (two new kwargs added):
```python
train_loader, val_loader = create_dataloaders(
    train_df, val_df, APTOS_TRAIN_IMAGES,
    batch_size=args.batch_size, num_workers=NUM_WORKERS,
    use_cache=use_cache, cache_dir=cache_dir,
    use_balanced_sampler=args.use_balanced_sampler,
    # Plan 08 additions:
    use_aug_balanced_dataset=args.use_aug_balanced_dataset,
    aug_target_count_per_class=args.aug_target_count_per_class,
)
if args.use_aug_balanced_dataset:
    print(
        f"  AugmentedBalancedDataset active: {len(train_loader.dataset)} training "
        f"samples (target {args.aug_target_count_per_class} per class)"
    )
```

---

### Task 1.4 — `train.py`: Update alpha weight logic

**File**: `src/train.py`
**Where**: The alpha weight computation block (~line 518), currently:
```python
if args.use_balanced_sampler or not args.use_class_weights:
    alpha_weights = torch.ones(NUM_CLASSES, device=device)
    ...
else:
    alpha_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
```

Replace with the extended three-branch logic:
```python
# Determine alpha weight strategy:
# - AugmentedBalancedDataset active + uniform flag: use [1,1,1,1,1]
#   Rationale: input distribution is already ~balanced via augmentation;
#   inverse-frequency alpha would over-correct and destabilise Grade 0.
# - WeightedRandomSampler active OR class weights disabled: use [1,1,1,1,1]
# - Default: inverse-frequency weights from training labels.
train_labels = torch.tensor(train_df["diagnosis"].values)

use_uniform_alpha = (
    (args.use_aug_balanced_dataset and args.aug_focal_alpha_uniform)
    or args.use_balanced_sampler
    or not args.use_class_weights
)
if use_uniform_alpha:
    alpha_weights = torch.ones(NUM_CLASSES, device=device)
    reason = (
        "AugmentedBalancedDataset + uniform flag"
        if (args.use_aug_balanced_dataset and args.aug_focal_alpha_uniform)
        else ("WeightedRandomSampler" if args.use_balanced_sampler else "class weights disabled")
    )
    print(f"  Alpha weights: uniform [1,1,1,1,1] ({reason})")
else:
    alpha_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
    print(f"  Class alpha weights: {alpha_weights.cpu().numpy().round(3)}")
```

---

### Task 1.5 — `train.py`: Update `hyperparams` dict

**File**: `src/train.py`
**Where**: The `hyperparams = { ... }` dict (~line 563).

Add these entries so the augmentation settings are captured in the run metrics JSON:
```python
hyperparams = {
    ...,  # existing entries unchanged
    # Plan 08
    "use_aug_balanced_dataset": args.use_aug_balanced_dataset,
    "aug_target_count_per_class": getattr(args, "aug_target_count_per_class", 800),
    "aug_focal_alpha_uniform": getattr(args, "aug_focal_alpha_uniform", True),
}
```

---

### Task 1.6 — Create `src/configs/gpu_aug_balanced.yaml`

**File**: `src/configs/gpu_aug_balanced.yaml` (new file)

```yaml
# Experiment: Augmentation-based class balancing, Plan 08
# Run C (full plan): quantity balancing + class-conditional augmentation intensity
# Baseline anchor: original baseline 20260403_120248 (Messidor-2 QWK 0.6233)
name: baseline_aug_balanced_fold0
description: >
  AugmentedBalancedDataset (target=800 per class) + class-conditional transforms
  (standard/medium/heavy) + uniform Focal Loss alpha + label smoothing 0.1.
  Original head — no deeper classifier to avoid Phase 7B memorization risk.

model_type: baseline
pretrained: true
dropout_rate: 0.5

epochs: 25
learning_rate: 0.0001
weight_decay: 0.0001
batch_size: 16
grad_clip_norm: 1.0
early_stopping_patience: 7

focal_gamma: 2.0
use_class_weights: true       # Overridden to uniform by aug_focal_alpha_uniform
label_smoothing: 0.1          # Retained from Phase 7B — beneficial for calibration

use_aug_balanced_dataset: true
aug_target_count_per_class: 800
aug_focal_alpha_uniform: true

use_balanced_sampler: false   # Explicitly off — mutually exclusive with above
classifier_hidden_dim: 0      # Original head — no deeper classifier
lr_warmup_epochs: 0           # Off — isolating augmentation effect

image_size: 512
num_workers: 4
fold: 0

device: cuda
use_amp: true

seed: 42
```

Also create a Run B config (quantity-only, for ablation):

```yaml
# File: src/configs/gpu_aug_balanced_quantity_only.yaml
# Run B (ablation): quantity balancing only, all grades use standard transform
# Isolates the effect of augmentation quantity from conditional intensity.
name: baseline_aug_quantity_only_fold0
description: >
  AugmentedBalancedDataset with standard-only augmentation for all grades.
  Tests whether quantity balancing alone (without conditional intensity) improves
  minority class recall. Run before Run C.

model_type: baseline
pretrained: true
dropout_rate: 0.5

epochs: 25
learning_rate: 0.0001
weight_decay: 0.0001
batch_size: 16
grad_clip_norm: 1.0
early_stopping_patience: 7

focal_gamma: 2.0
use_class_weights: true
label_smoothing: 0.1

use_aug_balanced_dataset: true
aug_target_count_per_class: 800
aug_focal_alpha_uniform: true

use_balanced_sampler: false
classifier_hidden_dim: 0
lr_warmup_epochs: 0

image_size: 512
num_workers: 4
fold: 0

device: cuda
use_amp: true

seed: 42
```

> **Note for Run B**: `gpu_aug_balanced_quantity_only.yaml` requires a one-line code change in `train.py` or dataset instantiation to override `grade_to_aug_level` to all `"standard"`. The simplest approach is a temporary inline override at dataset construction time — see Task 1.3 and add an `args.aug_all_standard` flag, OR just use `AUG_GRADE_LEVEL_OVERRIDE = {0:"standard",1:"standard",2:"standard",3:"standard",4:"standard"}` in config. The former is cleaner. This can be deferred and Run B implemented by momentarily hardcoding, then reverting for Run C.

---

## Phase 2 — Validation (No GPU Required)

### Task 2.1 — CPU smoke test

```bash
cd src
python train.py \
  --model baseline \
  --epochs 2 \
  --batch_size 2 \
  --fold 0 \
  --use_aug_balanced_dataset \
  --aug_target_count_per_class 800 \
  --aug_focal_alpha_uniform \
  --label_smoothing 0.1 \
  --use_class_weights
```

Or with the YAML config (after `experiment_config.py` is updated):
```bash
python train.py --config configs/gpu_aug_balanced.yaml --epochs 2 --batch_size 2
```

**Expected output**:
```
  AugmentedBalancedDataset active: 4644 training samples (target 800 per class)
  Alpha weights: uniform [1,1,1,1,1] (AugmentedBalancedDataset + uniform flag)
  Epoch 1/2
    Train — loss: x.xxxx  acc: x.xxxx
    Val   — loss: x.xxxx  acc: x.xxxx
    Per-class F1: NoDR=x.xxx  Mild=x.xxx  Mode=x.xxx  Seve=x.xxx  PDR =x.xxx
```

**Failure conditions to check**:
- Dataset length is not ~4,644 → `_build_index` bug.
- Alpha weights show inverse-frequency values → `use_uniform_alpha` logic wrong.
- `ValueError: mutually exclusive` raised with `use_balanced_sampler=False` → guard condition too broad.
- Import error for `AUG_TARGET_COUNT_PER_CLASS` → config import block not updated.

### Task 2.2 — Unit-level dataset checks

These can be run as one-off scripts or quick assertions in a Python REPL:

```python
# Quick verification script — run from repo root: python check_aug_dataset.py
import pandas as pd
import sys
sys.path.insert(0, "src")

from dataset import AugmentedBalancedDataset

df = pd.read_csv("data_split/train_label.csv")

# Use fold 0 training split (approximately — not exact fold assignment)
# For a fast check, just use the full df
ds = AugmentedBalancedDataset(
    df=df,
    image_dir="data_split/train_split",
    target_count_per_class=800,
    preprocess=False,   # skip preprocessing for speed
)

# Check 1: all grades reach target count
from collections import Counter
counts = Counter(grade for _, grade in ds._index)
for grade in range(5):
    assert counts[grade] >= 800, f"Grade {grade} has {counts[grade]} < 800"
    print(f"  Grade {grade}: {counts[grade]} samples ✓")

# Check 2: total length is correct
original_above_target = sum(
    c for c in df["diagnosis"].value_counts() if c >= 800
)
expected_min = original_above_target + 800 * sum(
    1 for c in df["diagnosis"].value_counts() if c < 800
)
print(f"  Total samples: {len(ds)}  (expected ≥ {expected_min})")

# Check 3: no grade is zero
assert all(counts.get(g, 0) > 0 for g in range(5)), "Some grade has 0 samples"

print("All checks passed.")
```

### Task 2.3 — Visual augmentation check

```python
# Visual spot-check — run in a Jupyter notebook or script
import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset import AugmentedBalancedDataset, DEFAULT_GRADE_TO_AUG_LEVEL

df = pd.read_csv("data_split/train_label.csv")

# Find a Grade 1 source image (worst performer)
grade1_ids = df[df["diagnosis"] == 1]["id_code"].tolist()
grade1_df = df[df["id_code"] == grade1_ids[0]].copy()
# Repeat the single image 6 times to get 6 augmented versions
repeated_df = pd.concat([grade1_df] * 6, ignore_index=True)

ds = AugmentedBalancedDataset(
    df=repeated_df,
    image_dir="data_split/train_split",
    target_count_per_class=1,   # Don't need expansion here
    preprocess=True,
)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    img_tensor, label = ds[i]
    # Unnormalise ImageNet normalisation for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).numpy()
    img = np.clip(img * std + mean, 0, 1)
    ax.imshow(img)
    ax.set_title(f"Heavy aug copy {i+1}")
    ax.axis("off")
plt.suptitle("Grade 1 — 6 augmented copies (should all look different)")
plt.tight_layout()
plt.savefig("outputs/figures/grade1_aug_check.png", dpi=100)
print("Saved: outputs/figures/grade1_aug_check.png")
```

**Expected**: All 6 images are visually distinct (different rotation, colour, possible elastic distortion). If they look identical, the transform is not being applied.

---

## Phase 3 — GPU Experiments

### Task 3.1 — Run B: Quantity-only ablation (GPU)

**Purpose**: Establish whether oversampling with *standard-only* augmentation (no class-conditional intensity) improves minority class recall. This isolates the quantity effect from the quality effect.

**How to run Run B cleanly**: Before starting, temporarily modify `AugmentedBalancedDataset.__init__` to override the grade level map to all `"standard"`:

```python
# Temporary override for Run B — revert before Run C
self.grade_to_aug_level = {g: "standard" for g in range(NUM_CLASSES)}
```

Or, better — add a `force_standard_aug: bool = False` parameter to `AugmentedBalancedDataset` and set it via a YAML flag.

**Run command (on GPU instance)**:
```bash
cd src
python train.py \
  --config configs/gpu_aug_balanced.yaml \
  --fold 0
# Override grade levels to all-standard for Run B:
# (manually set grade_to_aug_level or use force_standard_aug flag)
```

**Evaluate on Messidor-2 after training**:
```bash
python evaluate.py \
  --checkpoint outputs/checkpoints/baseline_aug_quantity_only_*_best.pth \
  --dataset messidor \
  --mc_passes 20 \
  --output_tag aug_run_b
```

**Decision gate**: If Run B Messidor-2 Grade 1 recall > 9.3% AND Grade 0 recall ≥ 85% → quantity effect confirmed → proceed to Run C.

---

### Task 3.2 — Run C: Full plan (GPU)

**Purpose**: Full `AugmentedBalancedDataset` with class-conditional transforms (heavy for Grades 1/2, medium for Grades 3/4, standard for Grade 0).

```bash
cd src
python train.py \
  --config configs/gpu_aug_balanced.yaml \
  --fold 0
```

**Evaluate on Messidor-2**:
```bash
python evaluate.py \
  --checkpoint outputs/checkpoints/baseline_aug_balanced_*_best.pth \
  --dataset messidor \
  --mc_passes 20 \
  --output_tag aug_run_c
```

**Success criteria**:
| Metric | Pass Condition |
|--------|---------------|
| Messidor-2 QWK | > 0.6233 (original baseline) |
| Messidor-2 Grade 1 Recall | > 9.3% |
| Messidor-2 Grade 2 Recall | > 16.1% |
| Messidor-2 Grade 0 Recall | ≥ 85.0% (guard rail — must NOT drop) |
| APTOS Val QWK | ≥ 0.85 (no catastrophic collapse) |

---

### Task 3.3 — Compare and record results

Build a comparison table after both runs complete:

```
| Metric                    | Original Baseline | Run B (qty only) | Run C (full plan) |
|---------------------------|-------------------|------------------|-------------------|
| Messidor-2 QWK            | 0.6233            | ?                | ?                 |
| Messidor-2 Accuracy       | 0.6279            | ?                | ?                 |
| Messidor-2 Grade 0 Recall | 0.942             | ?                | ?                 |
| Messidor-2 Grade 1 Recall | 0.093             | ?                | ?                 |
| Messidor-2 Grade 2 Recall | 0.161             | ?                | ?                 |
| Messidor-2 Grade 3 Recall | 0.560             | ?                | ?                 |
| Messidor-2 Grade 4 Recall | 0.400             | ?                | ?                 |
| APTOS Val QWK             | 0.9153            | ?                | ?                 |
| APTOS Val Macro F1        | 0.7326            | ?                | ?                 |
| Messidor-2 ECE            | 0.1155            | ?                | ?                 |
```

---

### Task 3.4 — Conditional: Multi-fold training

**Only proceed if**: Run C Messidor-2 QWK > 0.6233 AND Grade 0 recall ≥ 85%.

```bash
# Folds 1–4 with the winning config
for fold in 1 2 3 4; do
  python train.py --config configs/gpu_aug_balanced.yaml --fold $fold
done
```

---

## Phase 4 — Documentation Updates

### Task 4.1 — Update `docs/research-project.md`

- Section 11.4 (Pending Work): Mark `AugmentedBalancedDataset` as ✅ if implemented.
- Section 16 (Expected vs Actual Results): Add Run B / Run C rows to the Messidor-2 table.
- Section 16.6 (Key Lessons): Add lesson from this plan if results warrant it.

### Task 4.2 — Update `docs/result-report.md`

Add per-class recall comparison table (Run B vs Run C vs original baseline).

---

## Todo List

### Phase 0 — Infrastructure

- [ ] **T0.1** `src/config.py` — Add `USE_AUG_BALANCED_DATASET`, `AUG_TARGET_COUNT_PER_CLASS`, `AUG_FOCAL_ALPHA_UNIFORM`, `AUG_GRADE_LEVEL_OVERRIDE` constants after the Plan 07 block
- [ ] **T0.2** `src/dataset.py` — Add `import logging` to imports
- [ ] **T0.3** `src/dataset.py` — Add `AugLevel` type alias and `DEFAULT_GRADE_TO_AUG_LEVEL` dict after existing imports
- [ ] **T0.4** `src/dataset.py` — Add `USE_AUG_BALANCED_DATASET`, `AUG_TARGET_COUNT_PER_CLASS`, `AUG_FOCAL_ALPHA_UNIFORM`, `AUG_GRADE_LEVEL_OVERRIDE` to `from config import (...)` block
- [ ] **T0.5** `src/dataset.py` — Add `_build_standard_transform()` factory function after `get_val_transform()`
- [ ] **T0.6** `src/dataset.py` — Add `_build_medium_transform()` factory function
- [ ] **T0.7** `src/dataset.py` — Add `_build_heavy_transform()` factory function
- [ ] **T0.8** `src/dataset.py` — Add `AugmentedBalancedDataset` class with `_build_index()`, `__len__()`, `__getitem__()`
- [ ] **T0.9** `src/dataset.py` — Update `create_dataloaders()` signature: add `use_aug_balanced_dataset` and `aug_target_count_per_class` params
- [ ] **T0.10** `src/dataset.py` — Update `create_dataloaders()` body: add mutual-exclusion guard + `AugmentedBalancedDataset` branch

### Phase 1 — Training Integration

- [ ] **T1.1** `src/configs/experiment_config.py` — Add `USE_AUG_BALANCED_DATASET`, `AUG_TARGET_COUNT_PER_CLASS`, `AUG_FOCAL_ALPHA_UNIFORM` to imports from `config`
- [ ] **T1.2** `src/configs/experiment_config.py` — Add three new fields to `ExperimentConfig` dataclass after Plan 07 fields
- [ ] **T1.3** `src/configs/experiment_config.py` — Add three new assignments to `load_config()` function
- [ ] **T1.4** `src/train.py` — Add `--use_aug_balanced_dataset`, `--aug_target_count_per_class`, `--aug_focal_alpha_uniform` CLI arguments to `argparse` block
- [ ] **T1.5** `src/train.py` — Update `create_dataloaders(...)` call to pass the two new kwargs
- [ ] **T1.6** `src/train.py` — Replace alpha weight logic: extend `if args.use_balanced_sampler ...` block with three-way `use_uniform_alpha` check
- [ ] **T1.7** `src/train.py` — Add `use_aug_balanced_dataset`, `aug_target_count_per_class`, `aug_focal_alpha_uniform` to `hyperparams` dict
- [ ] **T1.8** Create `src/configs/gpu_aug_balanced.yaml` (Run C config)
- [ ] **T1.9** Create `src/configs/gpu_aug_balanced_quantity_only.yaml` (Run B config)

### Phase 2 — Validation

- [ ] **T2.1** Run CPU smoke test (2 epochs, batch_size=2, fold=0, `--use_aug_balanced_dataset`)
- [ ] **T2.2** Verify logged dataset size is ~4,644 (not 2,929) in smoke test output
- [ ] **T2.3** Verify alpha weights show `[1.0, 1.0, 1.0, 1.0, 1.0]` in smoke test output
- [ ] **T2.4** Run `check_aug_dataset.py` unit checks — all 5 grade counts ≥ 800
- [ ] **T2.5** Run visual augmentation check, save `grade1_aug_check.png`, visually confirm 6 copies differ
- [ ] **T2.6** Confirm `ValueError` is raised when both `use_aug_balanced_dataset=True` and `use_balanced_sampler=True`

### Phase 3 — GPU Experiments

- [ ] **T3.1** Spin up GPU instance (Vast.ai RTX 3060/3070/4070)
- [ ] **T3.2** Deploy updated codebase to GPU instance
- [ ] **T3.3** Run **Run B** (quantity-only): train fold 0 with all-standard transforms
- [ ] **T3.4** Evaluate Run B on Messidor-2 (T=20), save per-class recall
- [ ] **T3.5** Decision gate: if Run B Grade 0 recall < 85% → lower `aug_target_count_per_class` to 600 and re-run T3.3
- [ ] **T3.6** Run **Run C** (full plan): train fold 0 with class-conditional transforms
- [ ] **T3.7** Evaluate Run C on Messidor-2 (T=20), save per-class recall
- [ ] **T3.8** Decision gate: if Run C Grade 0 recall < 85% → investigate and adjust
- [ ] **T3.9** Build comparison table (original baseline vs Run B vs Run C) on all metrics
- [ ] **T3.10** *(Conditional on success)* Train remaining folds 1–4 with winning config

### Phase 4 — Documentation

- [ ] **T4.1** Update `docs/research-project.md` §11.4 pending work status
- [ ] **T4.2** Update `docs/research-project.md` §16 results tables with Run B / Run C data
- [ ] **T4.3** Update `docs/research-project.md` §16.6 key lessons if result warrants
- [ ] **T4.4** Add per-class recall comparison block to `docs/result-report.md`
- [ ] **T4.5** If Run C improves baseline: update §16.6 "Final Model Selection" to reflect new best model

---

## Quick Reference: Files Modified

| File | Type | Change |
|------|------|--------|
| `src/config.py` | Modify | +4 constants |
| `src/dataset.py` | Modify | +`import logging`, +type alias, +3 transform factories, +`AugmentedBalancedDataset`, +2 params to `create_dataloaders` |
| `src/configs/experiment_config.py` | Modify | +3 dataclass fields, +3 `load_config` assignments, +3 imports |
| `src/train.py` | Modify | +3 CLI args, +2 kwargs to `create_dataloaders` call, updated alpha logic, +3 hyperparams |
| `src/configs/gpu_aug_balanced.yaml` | New | Run C experiment config |
| `src/configs/gpu_aug_balanced_quantity_only.yaml` | New | Run B experiment config |

**Unchanged**: `model.py`, `loss.py`, `evaluate.py`, `preprocessing.py`, `temperature_scaling.py`, `threshold_tuning.py`. Validation and Messidor-2 inference paths are completely unaffected.
