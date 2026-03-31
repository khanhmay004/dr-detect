#!/usr/bin/env python3
"""
Compute cross-fold statistics from training history files.

Supports both old naming (model_fold0_history.json) and new timestamped
naming (model_YYYYMMDD_HHMMSS_fold0_history.json). For timestamped files,
uses the most recent run for each fold.

Usage:
    python src/compute_cross_fold_stats.py --model cbam_resnet50
    python src/compute_cross_fold_stats.py --model baseline_resnet50
    python src/compute_cross_fold_stats.py --model cbam_resnet50 --use_metrics
"""

import argparse
import json
import glob
import re
from pathlib import Path
import numpy as np

# Configuration
LOG_DIR = Path("outputs/logs")
RESULTS_DIR = Path("outputs/results")
N_FOLDS = 5


def find_history_files(model_name: str, fold: int) -> list:
    """Find all history files for a model/fold, sorted by timestamp (newest first)."""
    # Pattern: model_YYYYMMDD_HHMMSS_foldN_history.json (new) or model_foldN_history.json (old)
    patterns = [
        LOG_DIR / f"{model_name}_*_fold{fold}_history.json",  # timestamped
        LOG_DIR / f"{model_name}_fold{fold}_history.json",    # legacy
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(pattern)))

    # Sort by modification time (newest first)
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return files


def find_metrics_files(model_name: str, fold: int) -> list:
    """Find all metrics files for a model/fold, sorted by timestamp (newest first)."""
    patterns = [
        RESULTS_DIR / f"{model_name}_*_fold{fold}_metrics.json",  # timestamped
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(pattern)))

    # Sort by modification time (newest first)
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return files


def load_history(model_name: str, fold: int) -> dict:
    """Load training history for a single fold (most recent run)."""
    files = find_history_files(model_name, fold)
    if not files:
        print(f"  Warning: No history file found for {model_name} fold {fold}")
        return None

    path = files[0]  # Most recent
    print(f"  Loading: {Path(path).name}")
    with open(path) as f:
        return json.load(f)


def load_metrics(model_name: str, fold: int) -> dict:
    """Load detailed metrics JSON for a single fold (most recent run)."""
    files = find_metrics_files(model_name, fold)
    if not files:
        return None

    path = files[0]  # Most recent
    print(f"  Loading: {Path(path).name}")
    with open(path) as f:
        return json.load(f)


def compute_best_metrics_from_history(history: dict) -> dict:
    """Extract best epoch metrics from training history."""
    val_kappa = history["val_kappa"]
    best_idx = np.argmax(val_kappa)

    return {
        "best_epoch": int(best_idx + 1),
        "val_kappa": float(val_kappa[best_idx]),
        "val_acc": float(history["val_acc"][best_idx]),
        "val_auc": float(history["val_auc"][best_idx]),
        "val_sens": float(history["val_sens"][best_idx]) if "val_sens" in history else None,
        "val_spec": float(history["val_spec"][best_idx]) if "val_spec" in history else None,
        "val_loss": float(history["val_loss"][best_idx]),
        "train_acc": float(history["train_acc"][best_idx]),
        "train_loss": float(history["train_loss"][best_idx]),
    }


def extract_metrics_from_json(metrics: dict) -> dict:
    """Extract metrics from detailed metrics JSON."""
    best = metrics.get("best_metrics", {})
    run_info = metrics.get("run_info", {})

    return {
        "best_epoch": best.get("epoch", 0),
        "val_kappa": best.get("val_kappa", 0),
        "val_acc": best.get("val_acc", 0),
        "val_auc": best.get("val_auc", 0),
        "val_sens": best.get("val_sens"),
        "val_spec": best.get("val_spec"),
        "val_loss": best.get("val_loss", 0),
        "train_acc": best.get("train_acc", 0),
        "train_loss": best.get("train_loss", 0),
        "runtime_seconds": run_info.get("runtime_seconds", 0),
        "runtime_formatted": run_info.get("runtime_formatted", ""),
        "run_tag": run_info.get("run_tag", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute cross-fold statistics")
    parser.add_argument("--model", type=str, required=True,
                        choices=["baseline_resnet50", "cbam_resnet50"],
                        help="Model name prefix")
    parser.add_argument("--use_metrics", action="store_true",
                        help="Use detailed metrics JSON instead of history files")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Cross-Fold Statistics: {args.model.upper()}")
    print(f"{'=' * 60}\n")

    # Collect metrics from all folds
    fold_metrics = []
    total_runtime = 0

    for fold in range(N_FOLDS):
        if args.use_metrics:
            metrics_json = load_metrics(args.model, fold)
            if metrics_json:
                metrics = extract_metrics_from_json(metrics_json)
                metrics["fold"] = fold
                fold_metrics.append(metrics)
                total_runtime += metrics.get("runtime_seconds", 0)
                sens_str = f", Sens={metrics['val_sens']:.4f}" if metrics.get('val_sens') else ""
                spec_str = f", Spec={metrics['val_spec']:.4f}" if metrics.get('val_spec') else ""
                print(f"    Fold {fold}: QWK={metrics['val_kappa']:.4f}, "
                      f"Acc={metrics['val_acc']:.4f}, AUC={metrics['val_auc']:.4f}"
                      f"{sens_str}{spec_str}, "
                      f"Runtime={metrics.get('runtime_formatted', 'N/A')}")
        else:
            history = load_history(args.model, fold)
            if history is None:
                continue
            metrics = compute_best_metrics_from_history(history)
            metrics["fold"] = fold
            fold_metrics.append(metrics)
            sens_str = f", Sens={metrics['val_sens']:.4f}" if metrics.get('val_sens') else ""
            spec_str = f", Spec={metrics['val_spec']:.4f}" if metrics.get('val_spec') else ""
            print(f"    Fold {fold}: QWK={metrics['val_kappa']:.4f}, "
                  f"Acc={metrics['val_acc']:.4f}, AUC={metrics['val_auc']:.4f}"
                  f"{sens_str}{spec_str}")

    if len(fold_metrics) < 1:
        print("\n  Error: No folds found")
        return

    # Compute statistics
    kappas = [m["val_kappa"] for m in fold_metrics]
    accs = [m["val_acc"] for m in fold_metrics]
    aucs = [m["val_auc"] for m in fold_metrics]
    # Sens/Spec may be None for older history files
    sens_list = [m["val_sens"] for m in fold_metrics if m.get("val_sens") is not None]
    spec_list = [m["val_spec"] for m in fold_metrics if m.get("val_spec") is not None]

    print(f"\n{'-' * 60}")
    print(f"  SUMMARY ({len(fold_metrics)} folds)")
    print(f"{'-' * 60}")
    print(f"  Val QWK:      {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")
    print(f"  Val Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Val AUC:      {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    if sens_list:
        print(f"  Val Sens:     {np.mean(sens_list):.4f} +/- {np.std(sens_list):.4f}")
    if spec_list:
        print(f"  Val Spec:     {np.mean(spec_list):.4f} +/- {np.std(spec_list):.4f}")

    if total_runtime > 0:
        import time
        print(f"  Total Runtime: {time.strftime('%H:%M:%S', time.gmtime(total_runtime))}")

    print(f"{'-' * 60}\n")

    # Save to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "model": args.model,
        "timestamp": timestamp,
        "n_folds": len(fold_metrics),
        "per_fold": fold_metrics,
        "summary": {
            "val_kappa_mean": float(np.mean(kappas)),
            "val_kappa_std": float(np.std(kappas)),
            "val_acc_mean": float(np.mean(accs)),
            "val_acc_std": float(np.std(accs)),
            "val_auc_mean": float(np.mean(aucs)),
            "val_auc_std": float(np.std(aucs)),
        }
    }

    # Add sens/spec if available
    if sens_list:
        output["summary"]["val_sens_mean"] = float(np.mean(sens_list))
        output["summary"]["val_sens_std"] = float(np.std(sens_list))
    if spec_list:
        output["summary"]["val_spec_mean"] = float(np.mean(spec_list))
        output["summary"]["val_spec_std"] = float(np.std(spec_list))

    if total_runtime > 0:
        output["total_runtime_seconds"] = total_runtime

    output_path = RESULTS_DIR / f"{args.model}_crossfold_stats_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
