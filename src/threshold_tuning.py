"""Per-class decision threshold tuning for improved minority recall.

Finds threshold vector that maximises QWK (or Macro F1) on APTOS validation.
Apply at inference time: adjusted = mean_probs / thresholds; pred = argmax(adjusted).

Usage:
    python threshold_tuning.py \\
        --checkpoint "outputs/checkpoints/baseline*fold0_best.pth" \\
        --model baseline --fold 0 --metric qwk
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    APTOS_PROCESSED_DIR,
    APTOS_TRAIN_CSV,
    APTOS_TRAIN_IMAGES,
    BATCH_SIZE,
    DR_GRADES,
    IMAGE_SIZE,
    MC_DROPOUT_RATE,
    MC_INFERENCE_PASSES,
    NUM_CLASSES,
    NUM_WORKERS,
    RANDOM_SEED,
    RESULTS_DIR,
    USE_AMP,
    USE_PREPROCESSED_CACHE,
    seed_everything,
    setup_directories,
)
from dataset import DRDataset, get_train_val_split, get_val_transform
from model import create_baseline_model, create_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def mc_dropout_probs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_passes: int = MC_INFERENCE_PASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MC Dropout inference and return mean probabilities + labels.

    Args:
        model: Trained model.
        dataloader: DataLoader for validation set.
        device: Torch device.
        n_passes: Number of stochastic forward passes.

    Returns:
        Tuple of (mean_probs [N, C], labels [N]) as numpy arrays.
    """
    model.eval()
    amp_enabled = USE_AMP and device.type == "cuda"

    all_probs: list[torch.Tensor] = []
    all_labels: list[np.ndarray] = []

    pbar = tqdm(dataloader, desc=f"MC Inference (T={n_passes})")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)

        pass_probs = []
        for _ in range(n_passes):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
            probs = F.softmax(logits.float(), dim=1)
            pass_probs.append(probs.cpu())

        stacked = torch.stack(pass_probs, dim=0)
        mean_probs = stacked.mean(dim=0)
        all_probs.append(mean_probs)
        all_labels.append(labels.numpy())

    return (
        torch.cat(all_probs, dim=0).numpy(),
        np.concatenate(all_labels, axis=0),
    )


def tune_thresholds(
    mean_probs: np.ndarray,
    labels: np.ndarray,
    metric: str = "qwk",
) -> np.ndarray:
    """Find per-class threshold vector that maximises the target metric.

    Divides mean_probs by threshold vector before argmax:
        adjusted = mean_probs / thresholds
        prediction = argmax(adjusted)

    Args:
        mean_probs: Array of shape [N, C] with mean softmax probabilities.
        labels: Ground-truth labels [N].
        metric: Optimisation target — "qwk" or "macro_f1".

    Returns:
        Optimal threshold vector of shape [C], normalised so thresholds[0] = 1.0.

    Example:
        >>> probs = np.random.dirichlet([1]*5, size=100)
        >>> labels = np.random.randint(0, 5, 100)
        >>> thresholds = tune_thresholds(probs, labels, metric="qwk")
        >>> assert thresholds.shape == (5,)
    """
    num_classes = mean_probs.shape[1]

    def objective(log_thresholds: np.ndarray) -> float:
        thresholds = np.exp(log_thresholds)
        adjusted = mean_probs / thresholds[np.newaxis, :]
        predictions = adjusted.argmax(axis=1)
        if metric == "qwk":
            score = cohen_kappa_score(labels, predictions, weights="quadratic")
        elif metric == "macro_f1":
            score = f1_score(labels, predictions, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return -score

    initial = np.zeros(num_classes)
    result = minimize(
        objective, initial, method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-6}
    )

    optimal_thresholds = np.exp(result.x)
    optimal_thresholds = optimal_thresholds / optimal_thresholds[0]
    return optimal_thresholds


def compute_per_class_recall(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Compute recall for each class.

    Args:
        labels: Ground-truth labels [N].
        predictions: Predicted labels [N].
        num_classes: Number of classes.

    Returns:
        Array of per-class recall values [C].
    """
    recalls = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            recalls[c] = (predictions[mask] == c).sum() / mask.sum()
    return recalls


def print_comparison_table(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
    metric: str,
) -> dict[str, float]:
    """Print before/after comparison and return metrics dict.

    Args:
        labels: Ground-truth labels.
        probs: Mean probabilities [N, C].
        thresholds: Optimised threshold vector.
        metric: Target metric name.

    Returns:
        Dictionary with before/after metrics.
    """
    pred_before = probs.argmax(axis=1)
    adjusted = probs / thresholds[np.newaxis, :]
    pred_after = adjusted.argmax(axis=1)

    qwk_before = cohen_kappa_score(labels, pred_before, weights="quadratic")
    qwk_after = cohen_kappa_score(labels, pred_after, weights="quadratic")
    f1_before = f1_score(labels, pred_before, average="macro", zero_division=0)
    f1_after = f1_score(labels, pred_after, average="macro", zero_division=0)

    recall_before = compute_per_class_recall(labels, pred_before)
    recall_after = compute_per_class_recall(labels, pred_after)

    logger.info("\n" + "=" * 70)
    logger.info("  THRESHOLD TUNING RESULTS")
    logger.info("=" * 70)
    logger.info(f"  Optimisation target: {metric.upper()}")
    logger.info(f"  Thresholds: [{', '.join(f'{t:.3f}' for t in thresholds)}]")
    logger.info("-" * 70)
    logger.info(f"  {'Metric':<25} {'Before':>12} {'After':>12} {'Delta':>12}")
    logger.info("-" * 70)
    logger.info(f"  {'QWK':<25} {qwk_before:>12.4f} {qwk_after:>12.4f} {qwk_after-qwk_before:>+12.4f}")
    logger.info(f"  {'Macro F1':<25} {f1_before:>12.4f} {f1_after:>12.4f} {f1_after-f1_before:>+12.4f}")
    logger.info("-" * 70)
    logger.info("  Per-Class Recall:")
    for c in range(NUM_CLASSES):
        grade_name = DR_GRADES.get(c, f"Grade {c}")
        delta = recall_after[c] - recall_before[c]
        logger.info(f"    {c} ({grade_name:<15}) {recall_before[c]:>8.4f} {recall_after[c]:>12.4f} {delta:>+12.4f}")
    logger.info("=" * 70)

    return {
        "qwk_before": float(qwk_before),
        "qwk_after": float(qwk_after),
        "macro_f1_before": float(f1_before),
        "macro_f1_after": float(f1_after),
        "per_class_recall_before": recall_before.tolist(),
        "per_class_recall_after": recall_after.tolist(),
    }


def main() -> None:
    """CLI entry point for threshold tuning."""
    parser = argparse.ArgumentParser(description="Per-class threshold tuning")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint (wildcards supported)",
    )
    parser.add_argument(
        "--model", type=str, default="baseline",
        choices=["baseline", "cbam"],
        help="Model architecture",
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Validation fold to use for tuning",
    )
    parser.add_argument(
        "--metric", type=str, default="qwk",
        choices=["qwk", "macro_f1"],
        help="Target metric to maximise",
    )
    parser.add_argument(
        "--mc_passes", type=int, default=MC_INFERENCE_PASSES,
        help="Number of MC Dropout forward passes",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Batch size for inference",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if "*" in checkpoint_path or "?" in checkpoint_path:
        matches = sorted(glob.glob(checkpoint_path), key=os.path.getmtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"No checkpoint found matching: {checkpoint_path}")
        checkpoint_path = matches[0]
        logger.info(f"Resolved checkpoint: {checkpoint_path}")

    seed_everything(RANDOM_SEED)
    setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"\nLoading APTOS validation fold {args.fold} ...")
    full_df = pd.read_csv(APTOS_TRAIN_CSV)
    _, val_df = get_train_val_split(full_df, val_fold=args.fold)
    logger.info(f"  Validation images: {len(val_df)}")

    use_cache = USE_PREPROCESSED_CACHE and APTOS_PROCESSED_DIR.exists()
    cache_dir = APTOS_PROCESSED_DIR if use_cache else None

    val_dataset = DRDataset(
        df=val_df,
        image_dir=str(APTOS_TRAIN_IMAGES),
        transform=get_val_transform(IMAGE_SIZE),
        preprocess=not use_cache,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    logger.info(f"\nLoading {args.model.upper()} model ...")
    if args.model == "baseline":
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
    epoch = ckpt.get("epoch", "?")
    kappa = ckpt.get("best_kappa", "?")
    logger.info(f"  Checkpoint epoch: {epoch}  |  best kappa: {kappa}")

    logger.info(f"\nRunning MC Dropout inference (T={args.mc_passes}) ...")
    mean_probs, labels = mc_dropout_probs(model, val_loader, device, n_passes=args.mc_passes)
    logger.info(f"  Collected {len(labels)} samples")

    logger.info(f"\nTuning thresholds (metric={args.metric}) ...")
    thresholds = tune_thresholds(mean_probs, labels, metric=args.metric)

    metrics = print_comparison_table(labels, mean_probs, thresholds, args.metric)

    results = {
        "model": args.model,
        "checkpoint": Path(checkpoint_path).name,
        "fold": args.fold,
        "n_samples": len(labels),
        "metric": args.metric,
        "thresholds": thresholds.tolist(),
        **metrics,
    }

    output_path = RESULTS_DIR / f"{args.model}_thresholds.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results: {output_path}")

    logger.info("\n" + "-" * 50)
    logger.info("To apply these thresholds in evaluate.py, use:")
    threshold_str = ",".join(f"{t:.4f}" for t in thresholds)
    logger.info(f'  --thresholds "{threshold_str}"')
    logger.info("-" * 50)


if __name__ == "__main__":
    main()
