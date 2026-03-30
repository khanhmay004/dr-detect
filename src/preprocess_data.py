"""
Offline data preprocessing script.

Preprocesses all APTOS and Messidor-2 images with Ben Graham's method
and saves them to disk for faster training. This eliminates the need
to recompute preprocessing every epoch.

Performance impact:
    - On-the-fly: ~58,600 preprocessing ops over 20 epochs (~50-100 min overhead)
    - Offline: ~5 minutes once, then near-zero overhead

Usage:
    python preprocess_data.py                     # Process both datasets
    python preprocess_data.py --dataset aptos    # Process APTOS only
    python preprocess_data.py --dataset messidor # Process Messidor-2 only
    python preprocess_data.py --workers 8        # Use 8 parallel workers
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    APTOS_TRAIN_CSV,
    APTOS_TRAIN_IMAGES,
    MESSIDOR_CSV,
    MESSIDOR_IMAGES,
    PROCESSED_DIR,
    IMAGE_SIZE,
    FIGURES_DIR,
    RANDOM_SEED,
    seed_everything,
    setup_directories,
)
from preprocessing import ben_graham_preprocess, find_retina_circle


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================================
#  Preprocessing Statistics
# =========================================================================


@dataclass
class PreprocessingStats:
    """Statistics collected during preprocessing.

    Tracks success/failure counts, fallback usage, and radius distribution
    for quality monitoring and debugging.
    """

    total_images: int = 0
    successful: int = 0
    failed: int = 0
    fallback_used: int = 0
    mean_radius: float = 0.0
    min_radius: int = 0
    max_radius: int = 0
    radii: list = field(default_factory=list)


# =========================================================================
#  Single Image Processing
# =========================================================================


def preprocess_single_image(
    img_path: Path,
    output_path: Path,
    target_size: int = IMAGE_SIZE,
) -> Tuple[bool, int, bool, str | None]:
    """Preprocess a single image and save to output path.

    Applies Ben Graham preprocessing (circular crop + local color
    normalization) and saves as PNG.

    Args:
        img_path: Path to source image.
        output_path: Path to save preprocessed image.
        target_size: Output spatial resolution.

    Returns:
        Tuple of (success, radius, used_fallback, error_message).

    Example:
        >>> success, radius, fallback, err = preprocess_single_image(
        ...     Path("raw/img001.png"),
        ...     Path("processed/img001.png"),
        ... )
        >>> if success:
        ...     print(f"Processed with radius {radius}, fallback={fallback}")
    """
    if not img_path.exists():
        return (False, 0, False, f"File not found: {img_path}")

    image = cv2.imread(str(img_path))
    if image is None:
        return (False, 0, False, f"Failed to load: {img_path}")

    # Get preprocessing stats before full pipeline
    cx, cy, radius = find_retina_circle(image)

    # Check if fallback was used (full-image mode)
    # Fallback is used when no contours are found, returning half of min dimension
    image_height, image_width = image.shape[:2]
    expected_fallback_radius = min(image_width, image_height) // 2
    used_fallback = (radius == expected_fallback_radius)

    # Apply full preprocessing
    processed = ben_graham_preprocess(image, target_size)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save preprocessed image
    cv2.imwrite(str(output_path), processed)

    return (True, radius, used_fallback, None)


# =========================================================================
#  Dataset Processing
# =========================================================================


def preprocess_dataset(
    df: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
    id_column: str,
    extension: str = ".png",
    image_size: int = IMAGE_SIZE,
    num_workers: int = 4,
) -> PreprocessingStats:
    """Preprocess all images in a dataset.

    Processes images in parallel and saves to output directory with
    progress tracking and statistics collection.

    Args:
        df: DataFrame with image IDs.
        source_dir: Directory containing source images.
        output_dir: Directory to save preprocessed images.
        id_column: Column name containing image IDs.
        extension: Default image file extension.
        image_size: Target output size.
        num_workers: Number of parallel workers.

    Returns:
        PreprocessingStats with aggregated statistics.

    Example:
        >>> df = pd.read_csv("train.csv")
        >>> stats = preprocess_dataset(
        ...     df, Path("train_images"), Path("processed"),
        ...     id_column="id_code", num_workers=4,
        ... )
        >>> print(f"Processed {stats.successful}/{stats.total_images}")
    """
    stats = PreprocessingStats(total_images=len(df))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for _, row in df.iterrows():
        image_id = str(row[id_column])

        # Handle different image formats
        if Path(image_id).suffix:
            # Image ID already has extension
            source_path = source_dir / image_id
        else:
            source_path = source_dir / f"{image_id}{extension}"

        # Try common extensions if file doesn't exist
        if not source_path.exists():
            for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                candidate = source_dir / f"{image_id}{ext}"
                if candidate.exists():
                    source_path = candidate
                    break

        output_path = output_dir / f"{image_id}.png"
        tasks.append((source_path, output_path, image_size))

    # Process in parallel
    errors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(preprocess_single_image, *task): task
            for task in tasks
        }

        pbar = tqdm(
            as_completed(futures),
            total=len(tasks),
            desc="Preprocessing",
        )
        for future in pbar:
            task = futures[future]
            success, radius, used_fallback, error = future.result()

            if success:
                stats.successful += 1
                stats.radii.append(radius)
                if used_fallback:
                    stats.fallback_used += 1
            else:
                stats.failed += 1
                errors.append({"path": str(task[0]), "error": error})

    # Compute aggregate statistics
    if stats.radii:
        stats.mean_radius = float(np.mean(stats.radii))
        stats.min_radius = int(np.min(stats.radii))
        stats.max_radius = int(np.max(stats.radii))

    # Log summary
    logger.info("Preprocessing complete:")
    logger.info(f"  Total: {stats.total_images}")
    logger.info(f"  Successful: {stats.successful}")
    logger.info(f"  Failed: {stats.failed}")
    fallback_pct = 100 * stats.fallback_used / max(stats.successful, 1)
    logger.info(f"  Fallback used: {stats.fallback_used} ({fallback_pct:.1f}%)")
    logger.info(f"  Mean radius: {stats.mean_radius:.1f} px")
    logger.info(f"  Radius range: [{stats.min_radius}, {stats.max_radius}] px")

    # Save errors log if any
    if errors:
        error_path = output_dir / "preprocessing_errors.json"
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2)
        logger.warning(f"  Errors saved to: {error_path}")

    return stats


# =========================================================================
#  Visualization
# =========================================================================


def save_visualization_samples(
    df: pd.DataFrame,
    source_dir: Path,
    processed_dir: Path,
    id_column: str,
    output_path: Path,
    label_column: str | None = None,
    n_samples: int = 12,
) -> None:
    """Create before/after visualization grid.

    Generates a side-by-side comparison of original and preprocessed
    images, sampling from different DR grades if available.

    Args:
        df: DataFrame with image IDs.
        source_dir: Directory containing source images.
        processed_dir: Directory containing preprocessed images.
        id_column: Column name containing image IDs.
        output_path: Path to save visualization.
        label_column: Column containing DR grades (for stratified sampling).
        n_samples: Number of sample images to include.

    Example:
        >>> save_visualization_samples(
        ...     df, Path("raw"), Path("processed"),
        ...     id_column="id_code", label_column="diagnosis",
        ...     output_path=Path("figures/preprocessing_samples.png"),
        ... )
    """
    import matplotlib.pyplot as plt

    # Sample images from different grades if label column available
    if label_column and label_column in df.columns:
        samples = (
            df.groupby(label_column)
            .apply(
                lambda x: x.sample(
                    min(len(x), n_samples // 5 + 1),
                    random_state=RANDOM_SEED,
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        samples = samples.head(n_samples)
    else:
        samples = df.sample(min(len(df), n_samples), random_state=RANDOM_SEED)

    actual_n_samples = len(samples)
    fig, axes = plt.subplots(actual_n_samples, 2, figsize=(8, 2 * actual_n_samples))

    # Handle single sample case
    if actual_n_samples == 1:
        axes = axes.reshape(1, 2)

    for i, (_, row) in enumerate(samples.iterrows()):
        image_id = str(row[id_column])

        # Find original image
        source_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidate = source_dir / f"{image_id}{ext}"
            if candidate.exists():
                source_path = candidate
                break

        # Load original
        if source_path and source_path.exists():
            original = cv2.imread(str(source_path))
            if original is not None:
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                original = cv2.resize(original, (256, 256))
            else:
                original = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            original = np.zeros((256, 256, 3), dtype=np.uint8)

        # Load preprocessed
        processed_path = processed_dir / f"{image_id}.png"
        if processed_path.exists():
            processed = cv2.imread(str(processed_path))
            if processed is not None:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                processed = cv2.resize(processed, (256, 256))
            else:
                processed = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            processed = np.zeros((256, 256, 3), dtype=np.uint8)

        # Plot
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Original: {image_id}", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(processed)
        if label_column and label_column in row:
            label = row[label_column]
            axes[i, 1].set_title(f"Preprocessed (Grade {label})", fontsize=8)
        else:
            axes[i, 1].set_title("Preprocessed", fontsize=8)
        axes[i, 1].axis("off")

    plt.suptitle(
        "Ben Graham Preprocessing: Before vs After",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Visualization saved: {output_path}")


# =========================================================================
#  Main Entry Point
# =========================================================================


def main() -> None:
    """Main entry point for offline preprocessing."""
    parser = argparse.ArgumentParser(
        description="Offline data preprocessing for DR detection",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["aptos", "messidor", "all"],
        default="all",
        help="Which dataset to preprocess",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save before/after visualizations",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=IMAGE_SIZE,
        help="Target image size",
    )
    args = parser.parse_args()

    seed_everything(RANDOM_SEED)
    setup_directories()

    all_stats = {}

    # Process APTOS
    if args.dataset in ["aptos", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("Processing APTOS 2019 dataset")
        logger.info("=" * 60)

        aptos_df = pd.read_csv(APTOS_TRAIN_CSV)
        aptos_output = PROCESSED_DIR / "aptos"

        logger.info(f"  Source: {APTOS_TRAIN_IMAGES}")
        logger.info(f"  Output: {aptos_output}")
        logger.info(f"  Images: {len(aptos_df)}")

        stats = preprocess_dataset(
            df=aptos_df,
            source_dir=APTOS_TRAIN_IMAGES,
            output_dir=aptos_output,
            id_column="id_code",
            extension=".png",
            image_size=args.image_size,
            num_workers=args.workers,
        )
        all_stats["aptos"] = asdict(stats)
        all_stats["aptos"]["radii"] = None  # Don't save full list to JSON

        if args.visualize:
            save_visualization_samples(
                df=aptos_df,
                source_dir=APTOS_TRAIN_IMAGES,
                processed_dir=aptos_output,
                id_column="id_code",
                label_column="diagnosis",
                output_path=FIGURES_DIR / "preprocessing_aptos_samples.png",
            )

    # Process Messidor-2
    if args.dataset in ["messidor", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("Processing Messidor-2 dataset")
        logger.info("=" * 60)

        # Create DataFrame from image files directly (more robust)
        # The Messidor-2 images are PNG files in the IMAGES directory
        messidor_images = list(MESSIDOR_IMAGES.glob("*.png"))
        if not messidor_images:
            messidor_images = list(MESSIDOR_IMAGES.glob("*.tif"))
        if not messidor_images:
            messidor_images = list(MESSIDOR_IMAGES.glob("*.*"))

        messidor_df = pd.DataFrame({
            "image_id": [f.stem for f in messidor_images],
        })
        messidor_output = PROCESSED_DIR / "messidor2"

        logger.info(f"  Source: {MESSIDOR_IMAGES}")
        logger.info(f"  Output: {messidor_output}")
        logger.info(f"  Images: {len(messidor_df)}")

        stats = preprocess_dataset(
            df=messidor_df,
            source_dir=MESSIDOR_IMAGES,
            output_dir=messidor_output,
            id_column="image_id",
            extension=".png",
            image_size=args.image_size,
            num_workers=args.workers,
        )
        all_stats["messidor2"] = asdict(stats)
        all_stats["messidor2"]["radii"] = None

        if args.visualize:
            save_visualization_samples(
                df=messidor_df,
                source_dir=MESSIDOR_IMAGES,
                processed_dir=messidor_output,
                id_column="image_id",
                label_column=None,  # No labels in auto-generated df
                output_path=FIGURES_DIR / "preprocessing_messidor2_samples.png",
            )

    # Save preprocessing report
    report_path = PROCESSED_DIR / "preprocessing_report.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "image_size": args.image_size,
        "datasets": all_stats,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nPreprocessing report saved: {report_path}")


if __name__ == "__main__":
    main()
