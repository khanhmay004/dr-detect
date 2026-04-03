"""Post-hoc temperature scaling for probability calibration.

Learns a single scalar T on APTOS validation logits such that softmax(logits/T)
minimises NLL (equivalently, ECE). Does not change argmax predictions.

Usage:
    python temperature_scaling.py \\
        --checkpoint "outputs/checkpoints/baseline*fold0_best.pth" \\
        --model baseline --fold 0
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    APTOS_PROCESSED_DIR,
    APTOS_TRAIN_CSV,
    APTOS_TRAIN_IMAGES,
    BATCH_SIZE,
    FIGURES_DIR,
    IMAGE_SIZE,
    MC_DROPOUT_RATE,
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
from evaluate import compute_brier_score, compute_ece
from model import create_baseline_model, create_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def collect_val_logits(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run deterministic forward pass on validation set, collecting raw logits.

    Args:
        model: Trained model (with deterministic_mode available).
        val_loader: Validation DataLoader.
        device: Torch device.

    Returns:
        Tuple of (logits [N, C], labels [N]) on CPU.

    Example:
        >>> model = create_baseline_model(num_classes=5)
        >>> loader = DataLoader(...)  # doctest: +SKIP
        >>> logits, labels = collect_val_logits(model, loader, torch.device("cpu"))
        >>> assert logits.shape[1] == 5
    """
    model.eval()
    amp_enabled = USE_AMP and device.type == "cuda"
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with model.deterministic_mode():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled
            ):
                logits = model(images)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """Optimise temperature T to minimise NLL on validation logits.

    Args:
        logits: Raw logits [N, C].
        labels: Ground-truth labels [N].
        max_iter: Maximum LBFGS iterations.
        lr: Learning rate.

    Returns:
        Optimal temperature scalar.

    Example:
        >>> logits = torch.randn(100, 5)
        >>> labels = torch.randint(0, 5, (100,))
        >>> T = fit_temperature(logits, labels)
        >>> assert T > 0
    """
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled_logits = logits / temperature.clamp(min=0.01)
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().item())


def plot_reliability_comparison(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    n_bins: int = 15,
    save_path: Path | None = None,
) -> None:
    """Plot before/after reliability diagrams overlaid.

    Args:
        logits: Raw logits [N, C].
        labels: Ground-truth labels [N].
        temperature: Fitted temperature scalar.
        n_bins: Number of confidence bins.
        save_path: Optional path to save the figure.
    """
    labels_np = labels.numpy()

    probs_before = F.softmax(logits, dim=1).numpy()
    probs_after = F.softmax(logits / temperature, dim=1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, probs, title in [
        (axes[0], probs_before, f"Before (T=1.0, ECE={compute_ece(probs_before, labels_np, n_bins):.4f})"),
        (axes[1], probs_after, f"After (T={temperature:.3f}, ECE={compute_ece(probs_after, labels_np, n_bins):.4f})"),
    ]:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels_np).astype(float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for i in range(n_bins):
            in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
            if in_bin.sum() > 0:
                bin_accs.append(accuracies[in_bin].mean())
                bin_confs.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_accs.append(0)
                bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_counts.append(0)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.7, edgecolor="black")
        ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved reliability comparison: {save_path}")
    plt.close()


def main() -> None:
    """CLI entry point for temperature scaling."""
    parser = argparse.ArgumentParser(description="Post-hoc temperature scaling")
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
        help="Validation fold to use for fitting temperature",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--ece_bins", type=int, default=15,
        help="Number of bins for ECE computation",
    )
    parser.add_argument(
        "--classifier_hidden_dim", type=int, default=0,
        help="Classifier hidden dim (must match training config)",
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
            classifier_hidden_dim=args.classifier_hidden_dim,
        )
    else:
        model = create_model(
            num_classes=NUM_CLASSES,
            dropout_rate=MC_DROPOUT_RATE,
            pretrained=False,
            classifier_hidden_dim=args.classifier_hidden_dim,
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    epoch = ckpt.get("epoch", "?")
    kappa = ckpt.get("best_kappa", "?")
    logger.info(f"  Checkpoint epoch: {epoch}  |  best kappa: {kappa}")

    logger.info("\nCollecting validation logits ...")
    logits, labels = collect_val_logits(model, val_loader, device)
    logger.info(f"  Collected {len(labels)} samples")

    probs_before = F.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    ece_before = compute_ece(probs_before, labels_np, n_bins=args.ece_bins)
    brier_before = compute_brier_score(probs_before, labels_np, num_classes=NUM_CLASSES)
    logger.info(f"\nBefore temperature scaling:")
    logger.info(f"  ECE:   {ece_before:.4f}")
    logger.info(f"  Brier: {brier_before:.4f}")

    logger.info("\nFitting temperature ...")
    temperature = fit_temperature(logits, labels)
    logger.info(f"  Optimal T = {temperature:.4f}")

    probs_after = F.softmax(logits / temperature, dim=1).numpy()
    ece_after = compute_ece(probs_after, labels_np, n_bins=args.ece_bins)
    brier_after = compute_brier_score(probs_after, labels_np, num_classes=NUM_CLASSES)
    logger.info(f"\nAfter temperature scaling:")
    logger.info(f"  ECE:   {ece_after:.4f}")
    logger.info(f"  Brier: {brier_after:.4f}")

    results = {
        "model": args.model,
        "checkpoint": Path(checkpoint_path).name,
        "fold": args.fold,
        "n_samples": len(labels),
        "temperature": temperature,
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "ece_reduction": float(ece_before - ece_after),
        "brier_before": float(brier_before),
        "brier_after": float(brier_after),
        "brier_reduction": float(brier_before - brier_after),
    }

    output_path = RESULTS_DIR / f"{args.model}_temp_scaling.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results: {output_path}")

    fig_path = FIGURES_DIR / f"{args.model}_temp_scaling_reliability.png"
    plot_reliability_comparison(logits, labels, temperature, n_bins=args.ece_bins, save_path=fig_path)

    logger.info("\n" + "=" * 50)
    logger.info("  TEMPERATURE SCALING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Temperature:    {temperature:.4f}")
    logger.info(f"  ECE reduction:  {ece_before:.4f} → {ece_after:.4f} ({100*(ece_before-ece_after)/ece_before:.1f}% ↓)")
    logger.info(f"  Brier change:   {brier_before:.4f} → {brier_after:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
