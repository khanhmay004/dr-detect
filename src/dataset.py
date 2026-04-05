import logging
from collections import Counter
from typing import Literal, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from config import (
    APTOS_TRAIN_CSV, APTOS_TRAIN_IMAGES,
    MESSIDOR_IMAGES, MESSIDOR_CSV,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, NUM_WORKERS, RANDOM_SEED, N_FOLDS,
    APTOS_PROCESSED_DIR, MESSIDOR_PROCESSED_DIR, USE_PREPROCESSED_CACHE,
    NUM_CLASSES,
    USE_AUG_BALANCED_DATASET, AUG_TARGET_COUNT_PER_CLASS,
    AUG_FOCAL_ALPHA_UNIFORM, AUG_GRADE_LEVEL_OVERRIDE,
)
from preprocessing import ben_graham_preprocess


# =========================================================================
#  Class-conditional augmentation types (Plan 08)
# =========================================================================

AugLevel = Literal["standard", "medium", "heavy"]

DEFAULT_GRADE_TO_AUG_LEVEL: dict[int, AugLevel] = {
    0: "standard",   # 94.2% Messidor-2 recall — well-classified
    1: "heavy",      #  9.3% recall — worst performer
    2: "heavy",      # 16.1% recall — referable class, critical
    3: "medium",     # 56.0% recall — moderate failure
    4: "medium",     # 40.0% recall — significant failure
}


# =========================================================================
#  APTOS 2019 Dataset
# =========================================================================

class DRDataset(Dataset):
    """PyTorch Dataset for APTOS 2019 fundus images.

    Each ``__getitem__`` call:
      1. Loads the raw PNG via OpenCV (BGR).
      2. Applies Ben Graham preprocessing (crop + Gaussian-blur norm).
      3. Converts BGR → RGB.
      4. Applies optional albumentations transforms (augmentation + normalize).
      5. Returns ``(image_tensor, label)``.

    Args:
        df: DataFrame with columns ``id_code`` and ``diagnosis``.
        image_dir: Path to the directory containing ``.png`` images.
        transform: Albumentations ``Compose`` pipeline (or ``None``).
        preprocess: Whether to apply Ben Graham preprocessing.
        target_size: Spatial resolution after preprocessing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform: A.Compose | None = None,
        preprocess: bool = True,
        target_size: int = IMAGE_SIZE,
        use_cache: bool = False,
        cache_dir: Path | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.preprocess = preprocess
        self.target_size = target_size
        self.use_cache = use_cache
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_id = str(row["id_code"])

        cached_path = self.cache_dir / f"{image_id}.png" if self.cache_dir else None

        if self.use_cache and cached_path is not None and cached_path.exists():
            image = cv2.imread(str(cached_path))
            if image is None:
                raise ValueError(f"Failed to load cached image: {cached_path}")
        else:
            img_path = self.image_dir / f"{image_id}.png"
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Failed to load: {img_path}")
            if self.preprocess:
                image = ben_graham_preprocess(image, self.target_size)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = int(row["diagnosis"])

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(
                image.transpose(2, 0, 1).copy()
            ).float() / 255.0

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency weights for ``FocalLoss`` alpha."""
        counts = self.df["diagnosis"].value_counts().sort_index()
        total = len(self.df)
        weights = total / (len(counts) * counts)
        return torch.FloatTensor(weights.values)


# =========================================================================
#  Messidor-2 Dataset (external test set)
# =========================================================================

class MessidorDataset(Dataset):

    # Column names for the adjudicated grades CSV
    COL_IMAGE_ID = "image_id"
    COL_DR_GRADE = "adjudicated_dr_grade"
    COL_GRADABLE = "adjudicated_gradable"

    def __init__(
        self,
        csv_path: str = str(MESSIDOR_CSV),
        image_dir: str = str(MESSIDOR_IMAGES),
        transform: A.Compose | None = None,
        preprocess: bool = True,
        target_size: int = IMAGE_SIZE,
        labels_available: bool = True,
        use_cache: bool = False,
        cache_dir: Path | None = None,
        filter_gradable: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.preprocess = preprocess
        self.target_size = target_size
        self.labels_available = labels_available
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        # --- Detect CSV format and normalize column names ---
        self._normalize_columns()

        # --- Filter to gradable images only ---
        if filter_gradable and self.COL_GRADABLE in self.df.columns:
            n_before = len(self.df)
            self.df = self.df[self.df[self.COL_GRADABLE] == 1].reset_index(drop=True)
            n_after = len(self.df)
            if n_before != n_after:
                print(f"  Messidor-2: Filtered {n_before - n_after} ungradable "
                      f"images ({n_before} -> {n_after})")

        # --- Validate: check that at least some image files exist ---
        self._validate_image_availability()

    def _normalize_columns(self):
        """Detect CSV format and normalize to standard column names.

        Supports multiple CSV formats:
        - Adjudicated format: image_id, adjudicated_dr_grade, adjudicated_gradable
        - Legacy format: image, level (from EyePACS — should be flagged)
        """
        cols = set(self.df.columns)

        if "image_id" in cols and "adjudicated_dr_grade" in cols:
            # Standard adjudicated format — no changes needed
            pass
        elif "image" in cols and "level" in cols:
            # Legacy EyePACS format — rename for compatibility but warn
            print("  WARNING: CSV uses legacy 'image'/'level' column names.")
            print("    This may be the EyePACS trainLabels.csv — verify data source!")
            self.df = self.df.rename(columns={
                "image": "image_id",
                "level": "adjudicated_dr_grade",
            })
        else:
            available = list(self.df.columns)
            raise ValueError(
                f"Messidor-2 CSV has unexpected columns: {available}. "
                f"Expected 'image_id' + 'adjudicated_dr_grade' "
                f"or 'image' + 'level'."
            )

    def _validate_image_availability(self, sample_n: int = 5):
        """Check that at least some image IDs actually exist in IMAGES/."""
        sample = self.df[self.COL_IMAGE_ID].head(sample_n).tolist()
        found = 0
        for img_id in sample:
            try:
                self._find_image_path(str(img_id))
                found += 1
            except FileNotFoundError:
                pass
        if found == 0:
            raise RuntimeError(
                f"No Messidor-2 images found for the first {sample_n} IDs "
                f"in CSV. Sample IDs: {sample}. "
                f"Image directory: {self.image_dir}. "
                f"This likely means the CSV does not match the images — "
                f"check that you are using the correct Messidor-2 label file."
            )

    def __len__(self) -> int:
        return len(self.df)

    def _find_image_path(self, image_id: str) -> Path:
        """Resolve the full path — try common extensions."""
        for ext in [".tif", ".jpg", ".png", ".jpeg", ".JPG", ""]:
            candidate = self.image_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        candidate = self.image_dir / image_id
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Image not found for id '{image_id}' in {self.image_dir}"
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Returns ``(image, label, image_id)``."""
        row = self.df.iloc[idx]
        image_id = str(row[self.COL_IMAGE_ID])

        cached_path = self.cache_dir / f"{image_id}.png" if self.cache_dir else None

        if self.use_cache and cached_path is not None and cached_path.exists():
            image = cv2.imread(str(cached_path))
            if image is None:
                raise ValueError(f"Failed to load cached image: {cached_path}")
        else:
            img_path = self._find_image_path(image_id)
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Failed to load: {img_path}")
            if self.preprocess:
                image = ben_graham_preprocess(image, self.target_size)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read DR grade from adjudicated column
        if self.labels_available and self.COL_DR_GRADE in self.df.columns:
            label = int(row[self.COL_DR_GRADE])
        else:
            label = -1

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(
                image.transpose(2, 0, 1).copy()
            ).float() / 255.0

        return image, label, image_id


# =========================================================================
#  Albumentations transform pipelines
# =========================================================================

def get_train_transform(image_size: int = IMAGE_SIZE):
    """Training augmentations: heavy spatial + light color distortion."""
    if not HAS_ALBUMENTATIONS:
        return None
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            border_mode=cv2.BORDER_CONSTANT,
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


def get_val_transform(image_size: int = IMAGE_SIZE):
    """Validation / inference: resize + normalize only."""
    if not HAS_ALBUMENTATIONS:
        return None
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# =========================================================================
#  Class-conditional augmentation transform factories (Plan 08)
# =========================================================================

def _build_standard_transform(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Standard augmentation — identical to get_train_transform(). For Grade 0."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            border_mode=cv2.BORDER_CONSTANT,
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
    """Medium augmentation for Grades 3/4 — slightly less aggressive than heavy.

    Same conservative philosophy as heavy transform but with lower probabilities.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03, p=0.4
        ),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.85, 1.15),
            p=0.4,
        ),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            shear=(-5, 5),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.4,
        ),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.5), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _build_heavy_transform(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Conservative augmentation for Grades 1/2 — based on published DR paper.

    Preserves lesion structure: no elastic distortion, no coarse dropout,
    minimal hue shift (0.05 vs previous 0.15). Designed for subtle
    microaneurysm and hemorrhage patterns.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5
        ),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            p=0.5,
        ),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            shear=(-10, 10),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0), p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        A.Perspective(scale=(0.05, 0.2), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# =========================================================================
#  AugmentedBalancedDataset (Plan 08)
# =========================================================================

class AugmentedBalancedDataset(Dataset):
    """Training dataset that oversamples minority classes via augmentation.

    Builds an expanded sample index so each class reaches
    ``target_count_per_class``. Each oversampled repetition receives a freshly
    sampled stochastic augmentation — visually distinct from the source.

    Args:
        df: Training DataFrame with ``id_code`` and ``diagnosis`` columns.
        image_dir: Directory containing ``.png`` fundus images.
        target_count_per_class: Minimum samples per class after expansion.
        grade_to_aug_level: Grade → augmentation intensity mapping.
        image_size: Target spatial resolution.
        preprocess: Apply Ben Graham preprocessing.
        use_cache: Load from preprocessed cache.
        cache_dir: Path to cache directory.
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

        self._transforms: dict[AugLevel, A.Compose] = {
            "standard": _build_standard_transform(image_size),
            "medium":   _build_medium_transform(image_size),
            "heavy":    _build_heavy_transform(image_size),
        }

        self._index: list[tuple[int, int]] = self._build_index(
            target_count_per_class
        )

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
        """Build expanded (row_index, grade) list, cycling minority classes."""
        index: list[tuple[int, int]] = []
        for grade in range(NUM_CLASSES):
            grade_row_indices = self.df[
                self.df["diagnosis"] == grade
            ].index.tolist()

            n_source = len(grade_row_indices)
            if n_source == 0:
                logging.warning("Grade %d has 0 samples — skipped.", grade)
                continue

            n_target = max(n_source, target_count)

            for row_idx in grade_row_indices:
                index.append((row_idx, grade))

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

        aug_level: AugLevel = self.grade_to_aug_level.get(grade, "standard")
        transform = self._transforms[aug_level]
        augmented = transform(image=image)
        image_tensor: torch.Tensor = augmented["image"]  # [C, H, W]

        return image_tensor, grade


# =========================================================================
#  Train / Val split
# =========================================================================

def get_train_val_split(
    df: pd.DataFrame,
    val_fold: int = 0,
    n_folds: int = N_FOLDS,
    seed: int = RANDOM_SEED,
):
    """Stratified K-Fold split preserving class distribution."""
    df = df.copy()
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["diagnosis"])):
        df.loc[val_idx, "fold"] = fold_idx

    train_df = df[df["fold"] != val_fold].reset_index(drop=True)
    val_df = df[df["fold"] == val_fold].reset_index(drop=True)
    return train_df, val_df


# =========================================================================
#  Balanced Sampler
# =========================================================================

def make_balanced_sampler(
    labels: list[int] | np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> torch.utils.data.WeightedRandomSampler:
    """Create a WeightedRandomSampler that equalises class frequency per epoch.

    Each sample is weighted by the inverse of its class count, so minority
    classes are oversampled and majority classes are undersampled.

    Args:
        labels: Integer class labels for every sample in the dataset.
        num_classes: Number of distinct classes.

    Returns:
        WeightedRandomSampler configured for one epoch (len(labels) draws
        with replacement).

    Example:
        >>> labels = [0, 0, 0, 0, 1, 2]
        >>> sampler = make_balanced_sampler(labels, num_classes=3)
        >>> assert len(sampler) == 6
    """
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    sample_weights = 1.0 / class_counts[labels_tensor]
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(labels),
        replacement=True,
    )


# =========================================================================
#  DataLoader factories
# =========================================================================

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
    use_aug_balanced_dataset: bool = False,
    aug_target_count_per_class: int = AUG_TARGET_COUNT_PER_CLASS,
) -> Tuple[DataLoader, DataLoader]:
    """Create paired train/val DataLoaders for APTOS.

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        image_dir: Path to image directory.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        image_size: Target spatial resolution.
        preprocess: Apply Ben Graham preprocessing.
        use_cache: Load from preprocessed cache.
        cache_dir: Cache directory path.
        use_balanced_sampler: Use WeightedRandomSampler to equalise class
            frequency. Mutually exclusive with use_aug_balanced_dataset.
        use_aug_balanced_dataset: Use AugmentedBalancedDataset for training —
            oversamples minority classes via diverse augmentation. Mutually
            exclusive with use_balanced_sampler.
        aug_target_count_per_class: Target sample count per class after
            augmentation expansion. Ignored if use_aug_balanced_dataset=False.

    Returns:
        Tuple of (train_loader, val_loader).

    Raises:
        ValueError: If both use_aug_balanced_dataset and use_balanced_sampler
            are True.
    """
    if use_aug_balanced_dataset and use_balanced_sampler:
        raise ValueError(
            "use_aug_balanced_dataset and use_balanced_sampler are mutually "
            "exclusive — both attempt to balance class distribution. "
            "Use one or the other."
        )

    if use_aug_balanced_dataset:
        train_dataset = AugmentedBalancedDataset(
            df=train_df,
            image_dir=image_dir,
            target_count_per_class=aug_target_count_per_class,
            image_size=image_size,
            preprocess=preprocess,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
    else:
        train_dataset = DRDataset(
            df=train_df, image_dir=image_dir,
            transform=get_train_transform(image_size),
            preprocess=preprocess, target_size=image_size,
            use_cache=use_cache, cache_dir=cache_dir,
        )

    val_dataset = DRDataset(
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


def create_messidor_dataloader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    image_size: int = IMAGE_SIZE,
    labels_available: bool = True,
    use_cache: bool = False,
    cache_dir: Path | None = None,
) -> DataLoader:
    """Create a DataLoader for Messidor-2 inference."""
    dataset = MessidorDataset(
        transform=get_val_transform(image_size),
        preprocess=True,
        target_size=image_size,
        labels_available=labels_available,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    df = pd.read_csv(APTOS_TRAIN_CSV)
    train_df, val_df = get_train_val_split(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Class distribution (train):\n{train_df['diagnosis'].value_counts().sort_index()}")
