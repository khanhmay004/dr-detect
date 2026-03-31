"""
Uncertainty-aware evaluation on the Messidor-2 dataset.

This script loads a trained CBAM-ResNet50 checkpoint and runs MC Dropout
inference.  For each image it performs *T* stochastic forward passes and
computes:

- **Predictive mean distribution** p̄(y|x):  average softmax across T passes.
- **Predictive entropy** H[p̄]:  −Σ p̄_c · log(p̄_c).
  High entropy → the model is uncertain about this image.

The results are saved with timestamped, metadata-rich filenames::

    outputs/results/{model}_messidor2_{timestamp}_{T}T_{N}img_uncertainty.csv
    outputs/results/{model}_messidor2_{timestamp}_{T}T_{N}img_metrics.json

Usage::

    python evaluate.py --checkpoint outputs/checkpoints/cbam_resnet50_fold0_best.pth
    python evaluate.py --checkpoint best.pth --mc_passes 50
    python evaluate.py --checkpoint best.pth --model baseline --max_images 10 --mc_passes 3

    # Wildcard support (auto-selects most recent, works with/without timestamps):
    python evaluate.py --checkpoint "outputs/checkpoints/cbam_resnet50*fold0_best.pth"
    python evaluate.py --checkpoint "outputs/checkpoints/baseline*fold0_best.pth" --model baseline

MLOps notes
-----------
- ``model.eval()`` is called so BatchNorm uses its running statistics
  (stable predictions).  ``MCDropout`` is designed to remain active
  regardless of the model's training flag.
- We use ``torch.no_grad()`` because we never backpropagate during
  inference — this saves ~50 % memory by not storing activations.
- AMP autocast is used for the forward passes to keep VRAM usage low,
  but entropy is always computed in float32 for numerical precision.
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import (
    MESSIDOR_CSV, MESSIDOR_IMAGES,
    FIGURES_DIR, RESULTS_DIR, DR_GRADES, DR_COLORS,
    BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE,
    MC_DROPOUT_RATE, MC_INFERENCE_PASSES, NUM_CLASSES, USE_AMP,
    seed_everything, setup_directories, RANDOM_SEED,
)
from model import create_model, create_baseline_model
from dataset import MessidorDataset, get_val_transform


# =========================================================================
#  Core: MC Dropout uncertainty estimation
# =========================================================================

@torch.no_grad()
def mc_dropout_inference(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    n_passes: int = MC_INFERENCE_PASSES,
) -> dict:
    """Run T stochastic forward passes and compute predictive statistics.

    Args:
        model: Trained model (must contain ``MCDropout`` layers).
        dataloader: Messidor-2 DataLoader (yields image, label, image_id).
        device: CUDA or CPU.
        n_passes: Number of MC forward passes *T*.

    Returns:
        Dictionary with arrays:
          - image_ids:      list[str]
          - labels:         (N,)  ground-truth grades (-1 if unavailable)
          - mean_probs:     (N, C)  predictive mean distribution
          - entropy:        (N,)   predictive entropy
          - predictions:    (N,)   argmax of mean_probs
          - confidence:     (N,)   max of mean_probs
          - all_probs:      (N, T, C)  raw per-pass softmax (for analysis)
    """
    # model.eval() → BatchNorm uses running stats; MCDropout stays active
    model.eval()

    amp_enabled = USE_AMP and device.type == "cuda"

    all_image_ids = []
    all_labels = []
    all_stacked = []  # will collect (B, T, C) tensors

    pbar = tqdm(dataloader, desc=f"MC Inference (T={n_passes})")
    for images, labels, image_ids in pbar:
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)

        # --- T stochastic forward passes ---
        pass_probs = []
        for _ in range(n_passes):
            with torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled
            ):
                logits = model(images)
            # Softmax in float32 for numerical stability
            probs = F.softmax(logits.float(), dim=1)  # (B, C)
            pass_probs.append(probs.cpu())

        # Stack → (T, B, C) → permute → (B, T, C)
        stacked = torch.stack(pass_probs, dim=0).permute(1, 0, 2)
        all_stacked.append(stacked)

        all_image_ids.extend(image_ids)
        all_labels.extend(labels.numpy())

    # Concatenate batches → (N, T, C)
    all_stacked = torch.cat(all_stacked, dim=0)  # (N, T, C)
    N = all_stacked.size(0)

    # --- Predictive mean ---
    mean_probs = all_stacked.mean(dim=1)  # (N, C)

    # --- Predictive entropy  H = −Σ p̄_c · log(p̄_c) ---
    # Clamp to avoid log(0)
    entropy = -(mean_probs * torch.log(mean_probs.clamp(min=1e-10))).sum(dim=1)

    # --- Summary statistics ---
    confidence, predictions = mean_probs.max(dim=1)

    return {
        "image_ids": all_image_ids,
        "labels": np.array(all_labels),
        "mean_probs": mean_probs.numpy(),
        "entropy": entropy.numpy(),
        "predictions": predictions.numpy(),
        "confidence": confidence.numpy(),
        "all_probs": all_stacked.numpy(),  # (N, T, C) — for deeper analysis
    }


# =========================================================================
#  Metrics (when ground-truth labels are available)
# =========================================================================

def compute_metrics(labels: np.ndarray, predictions: np.ndarray, mean_probs: np.ndarray) -> dict:
    """Compute standard classification metrics including binary referable DR metrics.

    Args:
        labels: Ground truth grade labels, shape (N,).
        predictions: Predicted grade labels, shape (N,).
        mean_probs: Mean predicted probabilities, shape (N, C).

    Returns:
        Dictionary containing accuracy, quadratic_kappa, binary_referable_auc,
        binary_referable_sens, binary_referable_spec, and classification_report.
    """
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score,
        classification_report, roc_auc_score, confusion_matrix,
    )

    accuracy = accuracy_score(labels, predictions)
    kappa = cohen_kappa_score(labels, predictions, weights="quadratic")

    report = classification_report(
        labels, predictions,
        labels=list(DR_GRADES.keys()),  # Explicitly specify all 5 classes
        target_names=list(DR_GRADES.values()),
        output_dict=True, zero_division=0,
    )

    # Binary referable DR  (grade >= 2 = referable, < 2 = non-referable)
    binary_labels = (labels >= 2).astype(int)
    binary_preds = (predictions >= 2).astype(int)
    binary_probs = mean_probs[:, 2:].sum(axis=1)

    # AUC from probability scores
    try:
        binary_auc = roc_auc_score(binary_labels, binary_probs)
    except ValueError:
        binary_auc = 0.0

    # Sensitivity (recall of referable) & Specificity (recall of non-referable)
    # confusion_matrix returns [[TN, FP], [FN, TP]] for labels=[0,1]
    tn, fp, fn, tp = confusion_matrix(
        binary_labels, binary_preds, labels=[0, 1]
    ).ravel()
    binary_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    binary_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "quadratic_kappa": float(kappa),
        "binary_referable_auc": float(binary_auc),
        "binary_referable_sens": float(binary_sens),
        "binary_referable_spec": float(binary_spec),
        "classification_report": report,
    }


# =========================================================================
#  Visualization
# =========================================================================

def plot_uncertainty_histogram(entropy, predictions, save_path=None):
    """Histogram of predictive entropy, colored by predicted DR grade."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for grade, name in DR_GRADES.items():
        mask = predictions == grade
        if mask.any():
            ax.hist(
                entropy[mask], bins=30, alpha=0.6,
                color=DR_COLORS[grade], label=name, edgecolor="white",
            )

    ax.set_xlabel("Predictive Entropy", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Uncertainty Distribution - Messidor-2", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_confidence_vs_entropy(confidence, entropy, predictions, save_path=None):
    """Scatter plot: model confidence vs. predictive entropy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for grade, name in DR_GRADES.items():
        mask = predictions == grade
        if mask.any():
            ax.scatter(
                confidence[mask], entropy[mask],
                c=DR_COLORS[grade], label=name, alpha=0.5, s=20,
            )

    ax.set_xlabel("Confidence (max p)", fontsize=12)
    ax.set_ylabel("Predictive Entropy", fontsize=12)
    ax.set_title("Confidence vs. Uncertainty - Messidor-2",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# =========================================================================
#  Save per-image results
# =========================================================================

def save_results_csv(results: dict, save_path: Path):
    """Write one row per image with predictions and uncertainty."""
    rows = []
    for i, img_id in enumerate(results["image_ids"]):
        row = {
            "image_id": img_id,
            "true_grade": int(results["labels"][i]),
            "predicted_grade": int(results["predictions"][i]),
            "confidence": float(results["confidence"][i]),
            "entropy": float(results["entropy"][i]),
        }
        # Per-class probabilities
        for c in range(results["mean_probs"].shape[1]):
            row[f"p{c}"] = float(results["mean_probs"][i, c])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"  Results CSV: {save_path}  ({len(df)} images)")
    return df


# =========================================================================
#  Entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MC Dropout uncertainty inference on Messidor-2"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint (.pth file)",
    )
    parser.add_argument(
        "--model", type=str, default="cbam",
        choices=["baseline", "cbam"],
        help="Model architecture: 'baseline' (no CBAM) or 'cbam' (default)",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--mc_passes", type=int, default=MC_INFERENCE_PASSES,
                        help="Number of stochastic forward passes T")
    parser.add_argument("--no_labels", action="store_true",
                        help="Set if Messidor-2 CSV has no ground-truth grades")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit dataset to N images (for smoke testing)")
    args = parser.parse_args()

    # ---- Resolve checkpoint path (support wildcards) ----
    checkpoint_path = args.checkpoint
    if '*' in checkpoint_path or '?' in checkpoint_path:
        matches = sorted(glob.glob(checkpoint_path), key=os.path.getmtime, reverse=True)
        if not matches:
            raise FileNotFoundError(
                f"No checkpoint files found matching pattern: {checkpoint_path}"
            )
        checkpoint_path = matches[0]
        print(f"Resolved checkpoint pattern to: {checkpoint_path}")

    # Update args with resolved path
    args.checkpoint = checkpoint_path

    # ---- Setup ----
    seed_everything(RANDOM_SEED)
    setup_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset ----
    print("\nLoading Messidor-2 dataset ...")
    dataset = MessidorDataset(
        csv_path=str(MESSIDOR_CSV),
        image_dir=str(MESSIDOR_IMAGES),
        transform=get_val_transform(IMAGE_SIZE),
        preprocess=True,
        labels_available=not args.no_labels,
    )

    # Truncate for smoke testing
    if args.max_images and args.max_images < len(dataset):
        dataset.df = dataset.df.head(args.max_images).reset_index(drop=True)
        print(f"  [SMOKE TEST] Truncated to {len(dataset)} images")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"  Images: {len(dataset)}")
    if hasattr(dataset, 'COL_GRADABLE') and dataset.COL_GRADABLE in dataset.df.columns:
        print(f"  (filtered to gradable images only)")

    # ---- Build run tag for output file naming ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_images = len(dataset)
    run_tag = f"{args.model}_messidor2_{timestamp}_{args.mc_passes}T_{n_images}img"

    # ---- Model ----
    print(f"\nLoading {args.model.upper()} model from checkpoint ...")
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    epoch = ckpt.get("epoch", "?")
    kappa = ckpt.get("best_kappa", "?")
    print(f"  Checkpoint epoch: {epoch}  |  best kappa: {kappa}")

    # ---- MC Dropout Inference ----
    print(f"\nRunning MC Dropout inference (T = {args.mc_passes}) ...")
    results = mc_dropout_inference(
        model, dataloader, device, n_passes=args.mc_passes,
    )

    # ---- Results ----
    csv_path = RESULTS_DIR / f"{run_tag}_uncertainty.csv"
    save_results_csv(results, csv_path)

    # ---- Metrics (if labels available) ----
    if not args.no_labels and (results["labels"] >= 0).all():
        metrics = compute_metrics(
            results["labels"], results["predictions"], results["mean_probs"],
        )
        print(f"\n{'=' * 50}")
        print("  MESSIDOR-2 EVALUATION")
        print(f"{'=' * 50}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Quadratic kappa:   {metrics['quadratic_kappa']:.4f}")
        print(f"  Referable DR AUC:  {metrics['binary_referable_auc']:.4f}")
        print(f"  Referable DR Sens: {metrics['binary_referable_sens']:.4f}")
        print(f"  Referable DR Spec: {metrics['binary_referable_spec']:.4f}")

        # Save metrics JSON with run metadata
        metrics_path = RESULTS_DIR / f"{run_tag}_metrics.json"
        save_metrics = {
            "run_info": {
                "model": args.model,
                "checkpoint": str(Path(args.checkpoint).name),
                "checkpoint_epoch": epoch,
                "checkpoint_best_kappa": kappa,
                "dataset": "messidor2",
                "n_images": n_images,
                "mc_passes": args.mc_passes,
                "batch_size": args.batch_size,
                "image_size": IMAGE_SIZE,
                "device": str(device),
                "timestamp": datetime.now().isoformat(),
            },
        }
        for k, v in metrics.items():
            save_metrics[k] = v
        with open(metrics_path, "w") as f:
            json.dump(save_metrics, f, indent=2)
        print(f"\n  Metrics JSON: {metrics_path}")

    # ---- Plots ----
    print("\nGenerating plots ...")
    plot_uncertainty_histogram(
        results["entropy"], results["predictions"],
        save_path=FIGURES_DIR / f"{run_tag}_entropy_hist.png",
    )
    plot_confidence_vs_entropy(
        results["confidence"], results["entropy"], results["predictions"],
        save_path=FIGURES_DIR / f"{run_tag}_conf_vs_ent.png",
    )

    # ---- Summary statistics ----
    print(f"\n{'=' * 50}")
    print("  UNCERTAINTY SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Mean entropy:   {results['entropy'].mean():.4f}")
    print(f"  Median entropy: {np.median(results['entropy']):.4f}")
    print(f"  Max entropy:    {results['entropy'].max():.4f}")
    print(f"  Mean confidence:{results['confidence'].mean():.4f}")

    # Flag high-uncertainty predictions
    high_unc_mask = results["entropy"] > np.percentile(results["entropy"], 90)
    n_high = high_unc_mask.sum()
    print(f"\n  [!] {n_high} images ({100 * n_high / len(results['entropy']):.1f}%) "
          f"above 90th-percentile entropy — consider manual review.")

    print("\n[OK] Evaluation complete!")


if __name__ == "__main__":
    main()
