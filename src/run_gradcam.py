"""
Grad-CAM visualization CLI for DR detection models.

Generates lesion-focused heatmaps from trained checkpoints with support for:
    - Baseline and CBAM model architectures
    - Multiple target layers (layer3, layer4, cbam3, cbam4)
    - Case selection strategies (high_entropy, misclassified, etc.)
    - Side-by-side model comparison panels

Usage:
    # Basic usage with baseline model
    python run_gradcam.py \\
        --checkpoint "outputs/checkpoints/baseline_resnet50*fold0_best.pth" \\
        --model baseline \\
        --dataset messidor2 \\
        --num_images 10

    # CBAM model with high-entropy selection
    python run_gradcam.py \\
        --checkpoint "outputs/checkpoints/cbam_resnet50*fold0_best.pth" \\
        --model cbam \\
        --target_layer cbam4 \\
        --selection high_entropy \\
        --num_images 20

    # Bucketed analysis (all four buckets)
    python run_gradcam.py \\
        --checkpoint "outputs/checkpoints/baseline*fold0_best.pth" \\
        --model baseline \\
        --selection bucketed \\
        --n_per_bucket 5
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    FIGURES_DIR,
    RESULTS_DIR,
    MESSIDOR_CSV,
    MESSIDOR_IMAGES,
    APTOS_TRAIN_CSV,
    APTOS_TRAIN_IMAGES,
    APTOS_PROCESSED_DIR,
    MESSIDOR_PROCESSED_DIR,
    USE_PREPROCESSED_CACHE,
    IMAGE_SIZE,
    NUM_CLASSES,
    MC_DROPOUT_RATE,
    RANDOM_SEED,
    seed_everything,
    setup_directories,
)
from model import create_model, create_baseline_model
from dataset import (
    MessidorDataset,
    DRDataset,
    get_val_transform,
    get_train_val_split,
)
from interpretability import (
    GradCAM,
    get_target_module,
    denormalize_tensor,
    create_gradcam_panel,
    select_case_ids,
    select_by_criteria,
)


def resolve_checkpoint(pattern: str) -> str:
    """Resolve checkpoint path, supporting wildcards."""
    if "*" in pattern or "?" in pattern:
        matches = sorted(
            glob.glob(pattern),
            key=os.path.getmtime,
            reverse=True,
        )
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint files found matching: {pattern}"
            )
        return matches[0]
    return pattern


def load_model(
    checkpoint_path: str,
    model_type: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load model from checkpoint."""
    if model_type == "baseline":
        model = create_baseline_model(
            num_classes=NUM_CLASSES,
            dropout_rate=MC_DROPOUT_RATE,
            pretrained=False,
        )
    else:
        model = create_model(
            num_classes=NUM_CLASSES,
            dropout_rate=MC_DROPOUT_RATE,
            pretrained=False,
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def load_uncertainty_csv(results_dir: Path, model_type: str) -> pd.DataFrame:
    """Load the most recent uncertainty CSV for the model type."""
    pattern = results_dir / f"{model_type}_messidor2_*_uncertainty.csv"
    matches = sorted(
        glob.glob(str(pattern)),
        key=os.path.getmtime,
        reverse=True,
    )

    if not matches:
        raise FileNotFoundError(
            f"No uncertainty CSV found matching: {pattern}\n"
            "Run evaluate.py first to generate uncertainty outputs."
        )

    csv_path = matches[0]
    print(f"Using uncertainty CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    column_mapping = {
        "true_grade": "true_grade",
        "predicted_grade": "predicted_grade",
    }

    for old, new in column_mapping.items():
        if old not in df.columns:
            alt_names = {
                "true_grade": ["label", "true_label", "ground_truth"],
                "predicted_grade": ["prediction", "pred", "pred_grade"],
            }
            for alt in alt_names.get(old, []):
                if alt in df.columns:
                    df = df.rename(columns={alt: new})
                    break

    return df


def create_dataset(
    dataset_name: str,
    image_ids: list[str] | None = None,
):
    """Create dataset for Grad-CAM generation."""
    transform = get_val_transform(IMAGE_SIZE)

    if dataset_name == "messidor2":
        use_cache = USE_PREPROCESSED_CACHE and MESSIDOR_PROCESSED_DIR.exists()
        cache_dir = MESSIDOR_PROCESSED_DIR if use_cache else None

        dataset = MessidorDataset(
            csv_path=str(MESSIDOR_CSV),
            image_dir=str(MESSIDOR_IMAGES),
            transform=transform,
            preprocess=not use_cache,
            labels_available=True,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
    else:
        df = pd.read_csv(APTOS_TRAIN_CSV)
        _, val_df = get_train_val_split(df, val_fold=0)

        use_cache = USE_PREPROCESSED_CACHE and APTOS_PROCESSED_DIR.exists()
        cache_dir = APTOS_PROCESSED_DIR if use_cache else None

        dataset = DRDataset(
            df=val_df,
            image_dir=str(APTOS_TRAIN_IMAGES),
            transform=transform,
            preprocess=not use_cache,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )

    if image_ids is not None:
        if dataset_name == "messidor2":
            mask = dataset.df["image_id"].astype(str).isin(image_ids)
            dataset.df = dataset.df[mask].reset_index(drop=True)
        else:
            mask = dataset.df["id_code"].astype(str).isin(image_ids)
            dataset.df = dataset.df[mask].reset_index(drop=True)

    return dataset


def generate_gradcam_for_image(
    model: torch.nn.Module,
    gradcam: GradCAM,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, int, float]:
    """Generate Grad-CAM for a single image.

    Returns:
        Tuple of (cam_map, predicted_class, confidence).
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with model.deterministic_mode():
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = probs.max(dim=1)
            pred_class = int(pred_class.item())
            confidence = float(confidence.item())

    with model.deterministic_mode():
        cam = gradcam.generate(
            image_tensor,
            class_index=pred_class,
            upsample_size=(IMAGE_SIZE, IMAGE_SIZE),
        )

    return cam.numpy(), pred_class, confidence


def run_gradcam_analysis(args):
    """Main Grad-CAM analysis routine."""
    seed_everything(RANDOM_SEED)
    setup_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"Checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, args.model, device)
    print(f"Model: {args.model.upper()}")

    target_module = get_target_module(model, args.target_layer)
    print(f"Target layer: {args.target_layer}")

    gradcam = GradCAM(model, target_module)

    output_dir = FIGURES_DIR / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_ids = None

    if args.selection == "bucketed":
        uncertainty_df = load_uncertainty_csv(RESULTS_DIR, args.model)
        buckets = select_case_ids(uncertainty_df, n_per_bucket=args.n_per_bucket)

        all_image_ids = []
        for bucket_name, ids in buckets.items():
            print(f"  {bucket_name}: {len(ids)} images")
            all_image_ids.extend(ids)

        image_ids = list(set(all_image_ids))
        print(f"Selected {len(image_ids)} images for analysis")
    elif args.image_ids:
        image_ids = args.image_ids.split(",")
        print(f"Selected {len(image_ids)} images for analysis")
    elif args.selection == "random":
        image_ids = None
        print(f"Random selection: will sample {args.num_images} from dataset")
    else:
        uncertainty_df = load_uncertainty_csv(RESULTS_DIR, args.model)
        image_ids = select_by_criteria(
            uncertainty_df,
            selection=args.selection,
            n_images=args.num_images,
            seed=RANDOM_SEED,
        )
        print(f"Selected {len(image_ids)} images for analysis")

    dataset = create_dataset(args.dataset, image_ids=image_ids)

    if args.selection == "random" and image_ids is None:
        n_samples = min(args.num_images, len(dataset))
        indices = np.random.default_rng(RANDOM_SEED).choice(
            len(dataset), size=n_samples, replace=False
        )
        dataset.df = dataset.df.iloc[indices].reset_index(drop=True)

    print(f"Dataset: {args.dataset} ({len(dataset)} images)")

    if args.selection == "bucketed":
        uncertainty_df = load_uncertainty_csv(RESULTS_DIR, args.model)
    else:
        try:
            uncertainty_df = load_uncertainty_csv(RESULTS_DIR, args.model)
        except FileNotFoundError:
            uncertainty_df = None

    results = []
    for idx in range(len(dataset)):
        if args.dataset == "messidor2":
            image_tensor, true_grade, image_id = dataset[idx]
        else:
            image_tensor, true_grade = dataset[idx]
            image_id = str(dataset.df.iloc[idx]["id_code"])

        cam, pred_grade, confidence = generate_gradcam_for_image(
            model, gradcam, image_tensor, device
        )

        if uncertainty_df is not None:
            row = uncertainty_df[
                uncertainty_df["image_id"].astype(str) == str(image_id)
            ]
            if len(row) > 0:
                entropy = float(row.iloc[0]["entropy"])
            else:
                entropy = 0.0
        else:
            entropy = 0.0

        image_rgb = denormalize_tensor(image_tensor)

        fig = create_gradcam_panel(
            image=image_rgb,
            cam=cam,
            true_grade=true_grade,
            pred_grade=pred_grade,
            confidence=confidence,
            entropy=entropy,
            image_id=image_id,
            save_path=output_dir / f"{args.model}_{args.target_layer}_{image_id}.png",
        )
        plt.close(fig)

        results.append({
            "image_id": image_id,
            "model": args.model,
            "target_layer": args.target_layer,
            "true_grade": true_grade,
            "predicted_grade": pred_grade,
            "confidence": confidence,
            "entropy": entropy,
            "correct": true_grade == pred_grade,
        })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} images")

    gradcam.remove_hooks()

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / f"{args.model}_{args.target_layer}_{timestamp}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    if len(summary_df) > 0:
        n_correct = summary_df["correct"].sum()
        n_total = len(summary_df)
        print(f"\nResults: {n_correct}/{n_total} correct ({100*n_correct/n_total:.1f}%)")
        print(f"Mean confidence: {summary_df['confidence'].mean():.3f}")
        print(f"Mean entropy: {summary_df['entropy'].mean():.3f}")
    else:
        print("\nNo images processed.")

    print(f"\nGrad-CAM panels saved to: {output_dir}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualization for DR detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (supports wildcards)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["baseline", "cbam"],
        help="Model architecture",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="messidor2",
        choices=["messidor2", "aptos_val"],
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default="layer4",
        choices=["layer3", "layer4", "cbam3", "cbam4"],
        help="Target layer for Grad-CAM",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="high_entropy",
        choices=[
            "high_entropy",
            "low_entropy",
            "random",
            "misclassified",
            "false_negative",
            "bucketed",
        ],
        help="Case selection strategy",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=20,
        help="Number of images to analyze",
    )
    parser.add_argument(
        "--n_per_bucket",
        type=int,
        default=5,
        help="Images per bucket (for bucketed selection)",
    )
    parser.add_argument(
        "--image_ids",
        type=str,
        default=None,
        help="Comma-separated list of specific image IDs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/figures/gradcam)",
    )

    args = parser.parse_args()

    if args.target_layer.startswith("cbam") and args.model != "cbam":
        parser.error("CBAM layers only available for --model cbam")

    run_gradcam_analysis(args)


if __name__ == "__main__":
    main()
