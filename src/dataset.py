import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Tuple

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
)
from preprocessing import ben_graham_preprocess


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
) -> Tuple[DataLoader, DataLoader]:
    """Create paired train/val DataLoaders for APTOS."""
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
