# Implementation Plan: Phase 2 — CBAM Training + External Validation

> **Document Version**: 1.1
> **Created**: 2026-03-31
> **Updated**: 2026-03-31 — Added per-class metrics (F1/Recall/Precision per DR grade)
> **Scope**: Messidor-2 Baseline Evaluation + CBAM-ResNet50 Training (All 5 Folds)
> **Prerequisites**: Phase 1 baseline training complete (checkpoint: `dr-results/outputs/checkpoints/baseline_resnet50_fold0_best.pth`)

### Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-31 | Initial plan |
| 1.1 | 2026-03-31 | `train.py` `validate()` now returns per-class F1/recall/precision + macro F1; `fit()` prints per-class F1 table each epoch and logs to history; `_save_run_metrics()` persists per-class values in `*_metrics.json`; `compute_cross_fold_stats.py` computes and prints mean±std per-class table and saves to JSON |

---

## Executive Summary

Phase 2 completes the experimental pipeline by:

1. **Evaluating baseline ResNet-50 on Messidor-2** — External validation with MC Dropout uncertainty quantification
2. **Training CBAM-ResNet50 on fold 0** — Direct ablation comparison with baseline
3. **Training CBAM-ResNet50 on folds 1-4** — Cross-validation for robust statistics
4. **Computing cross-fold statistics** — Mean ± std for QWK, AUC, Accuracy, Sensitivity, Specificity, F1-macro, and **per-class F1/Recall/Precision** (5 DR grades)

**Expected Duration**: ~13 hours GPU time total on RTX 3090 (5 folds × 2.5 hrs training + ~20 min evaluations). Minimum useful run (fold 0 + both Messidor-2 evals): **~3 hours**.

---

## Table of Contents

1. [Phase 2.1: Messidor-2 Baseline Evaluation](#phase-21-messidor-2-baseline-evaluation)
2. [Phase 2.2: CBAM-ResNet50 Fold 0 Training](#phase-22-cbam-resnet50-fold-0-training)
3. [Phase 2.3: CBAM-ResNet50 Folds 1-4 Training](#phase-23-cbam-resnet50-folds-1-4-training)
4. [Phase 2.4: Cross-Fold Statistics](#phase-24-cross-fold-statistics)
5. [Vast.ai Setup Instructions](#vastai-setup-instructions)
6. [Expected Outputs](#expected-outputs)
7. [Detailed TODO Checklist](#detailed-todo-checklist)
8. [Verification Criteria](#verification-criteria)

---

## Phase 2.1: Messidor-2 Baseline Evaluation

### Purpose

Evaluate the trained baseline ResNet-50 on the external Messidor-2 dataset to:
- Measure domain generalization (APTOS → Messidor-2)
- Quantify prediction uncertainty using MC Dropout (T=20 passes)
- Establish baseline external validation metrics for CBAM comparison

### Prerequisites

- Baseline checkpoint: `dr-results/outputs/checkpoints/baseline_resnet50_fold0_best.pth`
- Messidor-2 dataset: `messidor-2/IMAGES/` + `messidor-2/messidor_data.csv`
- GPU with ≥8GB VRAM (evaluation is lighter than training)

### Command

```bash
# Full Messidor-2 evaluation with MC Dropout T=20
python src/evaluate.py \
    --checkpoint dr-results/outputs/checkpoints/baseline_resnet50_fold0_best.pth \
    --model baseline \
    --batch_size 16 \
    --mc_passes 20
```

### Smoke Test Command (verify setup first)

```bash
# Quick test with 10 images, T=3 passes
python src/evaluate.py \
    --checkpoint dr-results/outputs/checkpoints/baseline_resnet50_fold0_best.pth \
    --model baseline \
    --batch_size 4 \
    --mc_passes 3 \
    --max_images 10
```

### Expected Console Output

```
Device: cuda
Loading Messidor-2 dataset ...
  Images: 1744
  (filtered to gradable images only)

Loading BASELINE model from checkpoint ...
  Checkpoint epoch: 14  |  best kappa: 0.9088

Running MC Dropout inference (T = 20) ...
MC Inference (T=20): 100%|██████████| 109/109 [01:30<00:00]

  Results CSV: outputs/results/baseline_messidor2_YYYYMMDD_HHMMSS_20T_1744img_uncertainty.csv  (1744 images)

==================================================
  MESSIDOR-2 EVALUATION
==================================================
  Accuracy:          0.XXXX
  Quadratic kappa:   0.XXXX    # Expect 0.55-0.75 (5-15 points below APTOS val)
  Referable DR AUC:  0.XXXX    # Expect 0.80-0.92

==================================================
  UNCERTAINTY SUMMARY
==================================================
  Mean entropy:   0.XXXX
  Median entropy: 0.XXXX
  Max entropy:    X.XXXX
  Mean confidence: 0.XXXX
```

### Expected Outputs

```
outputs/
├── results/
│   ├── baseline_messidor2_YYYYMMDD_HHMMSS_20T_1744img_uncertainty.csv
│   └── baseline_messidor2_YYYYMMDD_HHMMSS_20T_1744img_metrics.json
└── figures/
    ├── baseline_messidor2_YYYYMMDD_HHMMSS_20T_1744img_entropy_hist.png
    └── baseline_messidor2_YYYYMMDD_HHMMSS_20T_1744img_conf_vs_ent.png
```

### Expected Metrics (Domain Shift)

| Metric | APTOS Val (Fold 0) | Expected Messidor-2 | Notes |
|--------|-------------------|---------------------|-------|
| **QWK** | 0.9088 | 0.55 - 0.75 | Domain shift expected |
| **Accuracy** | 84.45% | 60% - 75% | Different imaging devices |
| **Referable AUC** | 98.46% | 80% - 92% | Binary classification more robust |
| **Mean Entropy** | — | 0.3 - 0.8 | Higher = more uncertain |

---

## Phase 2.2: CBAM-ResNet50 Fold 0 Training

### Purpose

Train the CBAM-ResNet50 model on fold 0 with **identical hyperparameters** to baseline for valid ablation comparison. The only difference should be the CBAM attention modules.

### Command

```bash
python src/train.py \
    --model cbam \
    --epochs 20 \
    --batch_size 16 \
    --fold 0
```

### Configuration Verification

Before running, verify these match the baseline training:

| Parameter | Baseline (Phase 1) | CBAM (Phase 2) | Match? |
|-----------|-------------------|----------------|--------|
| epochs | 20 | 20 | ✓ |
| batch_size | 16 | 16 | ✓ |
| learning_rate | 1e-4 | 1e-4 | ✓ |
| weight_decay | 1e-4 | 1e-4 | ✓ |
| image_size | 512 | 512 | ✓ |
| focal_gamma | 2.0 | 2.0 | ✓ |
| dropout_rate | 0.5 | 0.5 | ✓ |
| grad_clip_norm | 1.0 | 1.0 | ✓ |
| AMP | True | True | ✓ |
| seed | 42 | 42 | ✓ |

### Expected Training Time

~2.5 hours on RTX 3090 (similar to baseline, CBAM adds ~8% overhead)

### Expected Console Output

```
Device: cuda (NVIDIA GeForce RTX 3090 - 25.3 GB)
Loading APTOS 2019 data ...
  Train: 2929  |  Val: 733
  Class weights α: [0.406, 2.121, 1.091, 3.763, 6.315]

Building CBAM model ...
  Parameters: 25,557,829  # ~2M more than baseline (23.5M)

=================================================================
  CBAM-ResNet50 - Fold 0
  Epochs: 1 -> 20  |  AMP: True  |  Device: cuda
=================================================================

Epoch 1/20 [Train]: 100%|██████████| 183/183 [02:30<00:00]
Epoch 1/20 [Val]: 100%|██████████| 46/46 [00:15<00:00]

  Epoch 1/20
    Train  — loss: 0.XXXX  acc: 0.XXXX
    Val    — loss: 0.XXXX  acc: 0.XXXX
    Val κ: 0.XXXX  AUC: 0.XXXX  Sens: 0.XXXX  Spec: 0.XXXX  F1-macro: 0.XXXX  LR: 1.00e-04
    Per-class F1:  No D=0.XXX  Mild=0.XXX  Mode=0.XXX  Seve=0.XXX  PDR =0.XXX
  ...
```

### Expected Outputs

```
outputs/
├── checkpoints/
│   ├── cbam_resnet50_fold0_best.pth   # ~290 MB
│   └── cbam_resnet50_fold0_last.pth   # ~290 MB
└── logs/
    └── cbam_resnet50_fold0_history.json
```

### Expected Metrics

| Metric | Baseline Fold 0 | Expected CBAM Fold 0 | Notes |
|--------|----------------|---------------------|-------|
| **Val QWK** | 0.9088 | 0.90 - 0.93 | Should match or exceed baseline |
| **Val Accuracy** | 84.45% | 83% - 87% | Attention may help minority classes |
| **Val AUC** | 98.46% | 98% - 99% | Binary classification ceiling |

---

## Phase 2.3: CBAM-ResNet50 Folds 1-4 Training

### Purpose

Train CBAM-ResNet50 on remaining folds for robust cross-validation statistics.

### Commands (Run Sequentially)

```bash
# Fold 1
python src/train.py --model cbam --epochs 20 --batch_size 16 --fold 1

# Fold 2
python src/train.py --model cbam --epochs 20 --batch_size 16 --fold 2

# Fold 3
python src/train.py --model cbam --epochs 20 --batch_size 16 --fold 3

# Fold 4
python src/train.py --model cbam --epochs 20 --batch_size 16 --fold 4
```

### Batch Script (run_all_folds.sh)

Create this script for automated sequential training:

```bash
#!/bin/bash
# run_all_folds.sh - Train CBAM on all 5 folds

set -e  # Exit on error

echo "=========================================="
echo "  CBAM-ResNet50 Cross-Validation Training"
echo "=========================================="

for fold in 0 1 2 3 4; do
    echo ""
    echo ">>> Starting Fold $fold at $(date)"
    echo ""

    python src/train.py \
        --model cbam \
        --epochs 20 \
        --batch_size 16 \
        --fold $fold \
        2>&1 | tee "outputs/logs/cbam_fold${fold}_training.log"

    echo ""
    echo ">>> Completed Fold $fold at $(date)"
    echo ""
done

echo "=========================================="
echo "  All folds complete!"
echo "=========================================="
```

### Expected Total Training Time

| Fold | Estimated Time |
|------|---------------|
| Fold 0 | ~2.5 hours |
| Fold 1 | ~2.5 hours |
| Fold 2 | ~2.5 hours |
| Fold 3 | ~2.5 hours |
| Fold 4 | ~2.5 hours |
| **Total** | **~12.5 hours** |

### Expected Outputs (All 5 Folds)

```
outputs/
├── checkpoints/
│   ├── cbam_resnet50_fold0_best.pth
│   ├── cbam_resnet50_fold0_last.pth
│   ├── cbam_resnet50_fold1_best.pth
│   ├── cbam_resnet50_fold1_last.pth
│   ├── cbam_resnet50_fold2_best.pth
│   ├── cbam_resnet50_fold2_last.pth
│   ├── cbam_resnet50_fold3_best.pth
│   ├── cbam_resnet50_fold3_last.pth
│   ├── cbam_resnet50_fold4_best.pth
│   └── cbam_resnet50_fold4_last.pth
└── logs/
    ├── cbam_resnet50_fold0_history.json
    ├── cbam_resnet50_fold1_history.json
    ├── cbam_resnet50_fold2_history.json
    ├── cbam_resnet50_fold3_history.json
    └── cbam_resnet50_fold4_history.json
```

---

## Phase 2.4: Cross-Fold Statistics

### Purpose

Aggregate results across all 5 folds to compute mean ± standard deviation for publication-quality results.

### Script: compute_cross_fold_stats.py

Create this script in `src/` directory:

```python
#!/usr/bin/env python3
"""
Compute cross-fold statistics from training history files.

Usage:
    python src/compute_cross_fold_stats.py --model cbam
    python src/compute_cross_fold_stats.py --model baseline
"""

import argparse
import json
from pathlib import Path
import numpy as np

# Configuration
LOG_DIR = Path("outputs/logs")
RESULTS_DIR = Path("outputs/results")
N_FOLDS = 5


def load_history(model_name: str, fold: int) -> dict:
    """Load training history for a single fold."""
    path = LOG_DIR / f"{model_name}_fold{fold}_history.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def compute_best_metrics(history: dict) -> dict:
    """Extract best epoch metrics from training history."""
    val_kappa = history["val_kappa"]
    best_idx = np.argmax(val_kappa)

    return {
        "best_epoch": best_idx + 1,
        "val_kappa": val_kappa[best_idx],
        "val_acc": history["val_acc"][best_idx],
        "val_auc": history["val_auc"][best_idx],
        "val_loss": history["val_loss"][best_idx],
        "train_acc": history["train_acc"][best_idx],
        "train_loss": history["train_loss"][best_idx],
    }


def main():
    parser = argparse.ArgumentParser(description="Compute cross-fold statistics")
    parser.add_argument("--model", type=str, required=True,
                        choices=["baseline_resnet50", "cbam_resnet50"],
                        help="Model name prefix")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Cross-Fold Statistics: {args.model.upper()}")
    print(f"{'=' * 60}\n")

    # Collect metrics from all folds
    fold_metrics = []
    for fold in range(N_FOLDS):
        history = load_history(args.model, fold)
        if history is None:
            continue
        metrics = compute_best_metrics(history)
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        print(f"  Fold {fold}: QWK={metrics['val_kappa']:.4f}, "
              f"Acc={metrics['val_acc']:.4f}, AUC={metrics['val_auc']:.4f}")

    if len(fold_metrics) < 2:
        print("\n  Error: Need at least 2 folds for statistics")
        return

    # Compute statistics
    kappas = [m["val_kappa"] for m in fold_metrics]
    accs = [m["val_acc"] for m in fold_metrics]
    aucs = [m["val_auc"] for m in fold_metrics]

    print(f"\n{'-' * 60}")
    print(f"  SUMMARY ({len(fold_metrics)} folds)")
    print(f"{'-' * 60}")
    print(f"  Val QWK:      {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")
    print(f"  Val Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Val AUC:      {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"{'-' * 60}\n")

    # Save to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": args.model,
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

    output_path = RESULTS_DIR / f"{args.model}_crossfold_stats.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
```

### Usage

```bash
# After all CBAM folds complete
python src/compute_cross_fold_stats.py --model cbam_resnet50

# For baseline (if all 5 folds were trained)
python src/compute_cross_fold_stats.py --model baseline_resnet50
```

### Expected Output

```
============================================================
  Cross-Fold Statistics: CBAM_RESNET50
============================================================

  Fold 0: QWK=0.9120, Acc=0.8512, AUC=0.9867, Sens=0.XXXX, Spec=0.XXXX
  Fold 1: QWK=0.9045, Acc=0.8423, AUC=0.9845, Sens=0.XXXX, Spec=0.XXXX
  Fold 2: QWK=0.9089, Acc=0.8467, AUC=0.9856, Sens=0.XXXX, Spec=0.XXXX
  Fold 3: QWK=0.9012, Acc=0.8389, AUC=0.9834, Sens=0.XXXX, Spec=0.XXXX
  Fold 4: QWK=0.9078, Acc=0.8456, AUC=0.9851, Sens=0.XXXX, Spec=0.XXXX

------------------------------------------------------------
  SUMMARY (5 folds)
------------------------------------------------------------
  Val QWK:      0.9069 +/- 0.0039
  Val Accuracy: 0.8449 +/- 0.0044
  Val AUC:      0.9851 +/- 0.0012
  Val Sens:     0.XXXX +/- 0.XXXX
  Val Spec:     0.XXXX +/- 0.XXXX
  Val F1-macro: 0.XXXX +/- 0.XXXX

  Per-class metrics (mean +/- std across 5 folds)
  --------------------------------------------------------------------------
  Grade               F1              Recall          Precision
  --------------------------------------------------------------------------
  No DR (0)    0.9XXX+/-0.00XX  0.9XXX+/-0.00XX  0.9XXX+/-0.00XX
  Mild (1)     0.5XXX+/-0.0XXX  0.5XXX+/-0.0XXX  0.5XXX+/-0.0XXX
  Moderate (2) 0.8XXX+/-0.00XX  0.8XXX+/-0.00XX  0.8XXX+/-0.00XX
  Severe (3)   0.4XXX+/-0.0XXX  0.4XXX+/-0.0XXX  0.4XXX+/-0.0XXX
  PDR (4)      0.7XXX+/-0.0XXX  0.7XXX+/-0.0XXX  0.7XXX+/-0.0XXX
  --------------------------------------------------------------------------
------------------------------------------------------------

  Saved: outputs/results/cbam_resnet50_crossfold_stats_YYYYMMDD_HHMMSS.json
```

---

## Vast.ai Setup Instructions

### Step 1: Instance Selection

**Recommended Instance**:
- GPU: RTX 3090 or RTX 4090 (24GB VRAM)
- vCPUs: 8+
- RAM: 32GB+
- Storage: 100GB+ SSD
- Docker Image: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`

**Cost Estimate**: ~$0.30-0.50/hour → ~$6-12 total for all experiments

### Step 2: Data Transfer

```bash
# On local machine: compress project + data
tar -czvf dr-detect-transfer.tar.gz \
    --exclude='outputs/checkpoints/*.pth' \
    --exclude='__pycache__' \
    --exclude='.git' \
    dr-detect/

# Upload to Vast.ai instance (via scp or web interface)
scp dr-detect-transfer.tar.gz root@<vast-ip>:/workspace/

# On Vast.ai: extract
cd /workspace
tar -xzvf dr-detect-transfer.tar.gz
cd dr-detect
```

### Step 3: Transfer Baseline Checkpoint

```bash
# Upload the trained baseline checkpoint
scp dr-results/outputs/checkpoints/baseline_resnet50_fold0_best.pth \
    root@<vast-ip>:/workspace/dr-detect/outputs/checkpoints/
```

### Step 4: Install Dependencies

```bash
# On Vast.ai instance
cd /workspace/dr-detect
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### Step 5: Verify Messidor-2 Dataset

```bash
# Check dataset structure
ls -la messidor-2/
ls messidor-2/IMAGES/ | head -5
wc -l messidor-2/messidor_data.csv

# Expected:
# - IMAGES/ directory with .tif files
# - messidor_data.csv with 1748 rows (header + 1747 images, 1744 gradable)
```

### Step 6: Run Experiments

```bash
# 1. Messidor-2 Baseline Evaluation (~5 min)
python src/evaluate.py \
    --checkpoint outputs/checkpoints/baseline_resnet50_fold0_best.pth \
    --model baseline \
    --batch_size 16 \
    --mc_passes 20

# 2. CBAM Fold 0 Training (~2.5 hours)
python src/train.py --model cbam --epochs 20 --batch_size 16 --fold 0

# 3. CBAM Messidor-2 Evaluation
python src/evaluate.py \
    --checkpoint "outputs/checkpoints/cbam_resnet50*fold0_best.pth" \
    --model cbam \
    --batch_size 16 \
    --mc_passes 20

# 4. (Optional) Remaining folds
for fold in 1 2 3 4; do
    python src/train.py --model cbam --epochs 20 --batch_size 16 --fold $fold
done

# 5. Cross-fold statistics
python src/compute_cross_fold_stats.py --model cbam_resnet50
```

### Step 7: Download Results

```bash
# On Vast.ai: compress outputs
cd /workspace/dr-detect
tar -czvf phase2-results.tar.gz outputs/

# Download to local machine
scp root@<vast-ip>:/workspace/dr-detect/phase2-results.tar.gz ./
```

---

## Expected Outputs

### New Naming Convention (Timestamped)

All outputs now include timestamps to avoid confusion between runs:

```
{model_name}_{YYYYMMDD_HHMMSS}_fold{N}_*.pth
{model_name}_{YYYYMMDD_HHMMSS}_fold{N}_*.json
```

Example: `cbam_resnet50_20260331_143052_fold0_best.pth`

### Directory Structure After Phase 2

```
outputs/
├── checkpoints/
│   ├── baseline_resnet50_20260331_HHMMSS_fold0_best.pth   # Timestamped
│   ├── baseline_resnet50_20260331_HHMMSS_fold0_last.pth
│   ├── cbam_resnet50_20260331_HHMMSS_fold0_best.pth       # NEW
│   ├── cbam_resnet50_20260331_HHMMSS_fold0_last.pth       # NEW
│   └── ... (folds 1-4 if trained)
├── logs/
│   ├── baseline_resnet50_20260331_HHMMSS_fold0_history.json
│   ├── cbam_resnet50_20260331_HHMMSS_fold0_history.json   # NEW
│   └── ... (folds 1-4 if trained)
├── results/
│   ├── baseline_resnet50_20260331_HHMMSS_fold0_metrics.json    # NEW - detailed run metrics
│   ├── cbam_resnet50_20260331_HHMMSS_fold0_metrics.json        # NEW
│   ├── baseline_messidor2_*_20T_*_uncertainty.csv              # Evaluation results
│   ├── baseline_messidor2_*_20T_*_metrics.json
│   ├── cbam_messidor2_*_20T_*_uncertainty.csv
│   ├── cbam_messidor2_*_20T_*_metrics.json
│   └── cbam_resnet50_crossfold_stats.json                      # Cross-fold aggregation
└── figures/
    ├── baseline_messidor2_*_entropy_hist.png
    ├── baseline_messidor2_*_conf_vs_ent.png
    ├── cbam_messidor2_*_entropy_hist.png
    └── cbam_messidor2_*_conf_vs_ent.png
```

### Training Metrics JSON Structure

Each training run produces a detailed `{run_tag}_metrics.json`:

```json
{
  "run_info": {
    "run_tag": "cbam_resnet50_20260331_143052_fold0",
    "model_name": "cbam_resnet50",
    "fold": 0,
    "timestamp": "20260331_143052",
    "start_time": "2026-03-31T14:30:52",
    "end_time": "2026-03-31T17:05:23",
    "runtime_seconds": 9271,
    "runtime_formatted": "02:34:31"
  },
  "best_metrics": {
    "epoch": 14,
    "val_kappa": 0.9120,
    "val_acc": 0.8512,
    "val_auc": 0.9867,
    "val_loss": 0.4823,
    "train_acc": 0.8945,
    "train_loss": 0.1892
  },
  "final_metrics": {
    "epoch": 19,
    "val_kappa": 0.9045,
    "val_acc": 0.8423,
    "val_auc": 0.9845,
    "val_loss": 0.5234,
    "train_acc": 0.9234,
    "train_loss": 0.1234
  },
  "hyperparameters": {
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "focal_gamma": 2.0,
    "image_size": 512,
    "dropout_rate": 0.5,
    "grad_clip_norm": 1.0,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "seed": 42,
    "n_folds": 5,
    "train_size": 2929,
    "val_size": 733,
    "total_params": 25557829
  },
  "training_config": {
    "epochs_requested": 20,
    "epochs_completed": 19,
    "early_stopped": true,
    "device": "cuda",
    "amp_enabled": true,
    "grad_clip_norm": 1.0
  },
  "checkpoints": {
    "best": "cbam_resnet50_20260331_143052_fold0_best.pth",
    "last": "cbam_resnet50_20260331_143052_fold0_last.pth"
  }
}
```

### Key Metrics to Record

| Experiment | Metric | Expected Value |
|------------|--------|----------------|
| **Baseline Messidor-2** | QWK | 0.55 - 0.75 |
| **Baseline Messidor-2** | Accuracy | 60% - 75% |
| **Baseline Messidor-2** | Referable AUC | 80% - 92% |
| **CBAM Fold 0** | Val QWK | 0.90 - 0.93 |
| **CBAM Fold 0** | Val Accuracy | 83% - 87% |
| **CBAM Messidor-2** | QWK | 0.55 - 0.78 |
| **CBAM Cross-Fold** | QWK (mean±std) | 0.90 ± 0.02 |

---

## Detailed TODO Checklist

### Phase 2.1: Messidor-2 Baseline Evaluation

| # | Task | Status | Depends On | Notes |
|---|------|--------|------------|-------|
| 2.1.1 | Verify baseline checkpoint exists | `[ ]` | Phase 1 | `dr-results/outputs/checkpoints/baseline_resnet50_fold0_best.pth` |
| 2.1.2 | Transfer checkpoint to Vast.ai | `[ ]` | 2.1.1 | `outputs/checkpoints/` |
| 2.1.3 | Verify Messidor-2 dataset | `[ ]` | — | `messidor-2/IMAGES/` + CSV |
| 2.1.4 | Run evaluation smoke test | `[ ]` | 2.1.2, 2.1.3 | `--max_images 10 --mc_passes 3` |
| 2.1.5 | Run full Messidor-2 evaluation | `[ ]` | 2.1.4 | `--mc_passes 20` |
| 2.1.6 | Record baseline external metrics | `[ ]` | 2.1.5 | QWK, AUC, accuracy |
| 2.1.7 | Save uncertainty CSV | `[ ]` | 2.1.5 | Per-image predictions |
| 2.1.8 | Generate uncertainty figures | `[ ]` | 2.1.5 | Histogram + scatter |
| 2.1.9 | Compute entropy statistics | `[ ]` | 2.1.5 | Mean, median, 90th percentile |

### Phase 2.2: CBAM-ResNet50 Fold 0 Training

| # | Task | Status | Depends On | Notes |
|---|------|--------|------------|-------|
| 2.2.1 | Verify hyperparameters match baseline | `[ ]` | — | Same epochs, batch_size, lr, etc. |
| 2.2.2 | Run CBAM fold 0 training | `[ ]` | 2.2.1 | `--model cbam --fold 0` |
| 2.2.3 | Monitor training curves | `[ ]` | 2.2.2 | Check for convergence |
| 2.2.4 | Record CBAM fold 0 metrics | `[ ]` | 2.2.2 | QWK, AUC, accuracy |
| 2.2.5 | Compare CBAM vs baseline fold 0 | `[ ]` | 2.2.4 | Ablation comparison |
| 2.2.6 | Run CBAM Messidor-2 evaluation | `[ ]` | 2.2.2 | External validation |
| 2.2.7 | Compare external metrics | `[ ]` | 2.2.6, 2.1.6 | CBAM vs baseline on Messidor-2 |

### Phase 2.3: CBAM-ResNet50 Folds 1-4 Training (Optional)

| # | Task | Status | Depends On | Notes |
|---|------|--------|------------|-------|
| 2.3.1 | Create run_all_folds.sh script | `[ ]` | — | Automated batch training |
| 2.3.2 | Run CBAM fold 1 training | `[ ]` | 2.2.2 | ~2.5 hours |
| 2.3.3 | Run CBAM fold 2 training | `[ ]` | 2.3.2 | ~2.5 hours |
| 2.3.4 | Run CBAM fold 3 training | `[ ]` | 2.3.3 | ~2.5 hours |
| 2.3.5 | Run CBAM fold 4 training | `[ ]` | 2.3.4 | ~2.5 hours |
| 2.3.6 | Verify all 5 checkpoints saved | `[ ]` | 2.3.5 | Check file sizes |

### Phase 2.4: Cross-Fold Statistics (Optional)

| # | Task | Status | Depends On | Notes |
|---|------|--------|------------|-------|
| 2.4.1 | Create compute_cross_fold_stats.py | `[ ]` | — | New script |
| 2.4.2 | Run cross-fold aggregation | `[ ]` | 2.3.6, 2.4.1 | All histories required |
| 2.4.3 | Record mean ± std metrics | `[ ]` | 2.4.2 | For publication |
| 2.4.4 | Save cross-fold JSON | `[ ]` | 2.4.2 | `cbam_resnet50_crossfold_stats.json` |

### Phase 2.5: Results Download & Documentation

| # | Task | Status | Depends On | Notes |
|---|------|--------|------------|-------|
| 2.5.1 | Compress all outputs | `[ ]` | 2.2.7 (min) | `tar -czvf phase2-results.tar.gz outputs/` |
| 2.5.2 | Download from Vast.ai | `[ ]` | 2.5.1 | scp to local |
| 2.5.3 | Update research-project.md | `[ ]` | 2.5.2 | Document results |
| 2.5.4 | Update plans/02-phase1-3.md | `[ ]` | 2.5.2 | Mark Phase 2 complete |

---

## Verification Criteria

### Phase 2.1 Success Criteria

- [ ] Messidor-2 evaluation completes without errors
- [ ] All 1744 gradable images processed (or appropriate subset)
- [ ] QWK computed (expect 0.55-0.75 for domain shift)
- [ ] Uncertainty CSV contains per-image entropy values
- [ ] Figures saved (entropy histogram, confidence vs entropy)

### Phase 2.2 Success Criteria

- [ ] CBAM training completes all 20 epochs (or early stops)
- [ ] Val QWK ≥ 0.85 (sanity check)
- [ ] Val QWK comparable to or better than baseline (0.9088)
- [ ] Checkpoint file size ~290 MB (larger than baseline due to CBAM)
- [ ] Training curves show stable convergence

### Phase 2.3 Success Criteria (Optional)

- [ ] All 5 folds complete without errors
- [ ] Consistent performance across folds (±2-3% variation expected)
- [ ] All checkpoints and histories saved

### Phase 2.4 Success Criteria (Optional)

- [ ] Cross-fold statistics computed correctly
- [ ] Standard deviation < 0.05 for QWK (stable model)
- [ ] JSON summary saved for publication

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Messidor-2 path resolution fails | Low | Medium | Test with smoke test first; verify extensions |
| CBAM underperforms baseline | Medium | High | Report honestly; check for bugs in CBAM modules |
| Training diverges (NaN loss) | Low | High | Gradient clipping in place; monitor early epochs |
| Vast.ai instance interrupted | Medium | Medium | Save checkpoints frequently; use preemptible wisely |
| Messidor-2 label format mismatch | Low | Medium | evaluate.py handles multiple formats |
| Memory OOM on evaluation | Low | Medium | Reduce batch_size if needed |

---

## Quick Reference Commands

```bash
# ========================================
# Phase 2.1: Messidor-2 Baseline Evaluation
# ========================================

# Smoke test (quick)
python src/evaluate.py \
    --checkpoint "outputs/checkpoints/baseline_resnet50*fold0_best.pth" \
    --model baseline --batch_size 4 --mc_passes 3 --max_images 10

# Full evaluation (wildcard auto-selects most recent)
python src/evaluate.py \
    --checkpoint "outputs/checkpoints/baseline_resnet50*fold0_best.pth" \
    --model baseline --batch_size 16 --mc_passes 20


# ========================================
# Phase 2.2: CBAM Fold 0 Training
# ========================================

python src/train.py --model cbam --epochs 20 --batch_size 16 --fold 0

# CBAM Messidor-2 evaluation (wildcard auto-selects most recent)
python src/evaluate.py \
    --checkpoint "outputs/checkpoints/cbam_resnet50*fold0_best.pth" \
    --model cbam --batch_size 16 --mc_passes 20


# ========================================
# Phase 2.3: CBAM Folds 1-4 (Optional)
# ========================================

for fold in 1 2 3 4; do
    python src/train.py --model cbam --epochs 20 --batch_size 16 --fold $fold
done


# ========================================
# Phase 2.4: Cross-Fold Statistics
# ========================================

# Using history files
python src/compute_cross_fold_stats.py --model cbam_resnet50

# Using detailed metrics JSON (includes runtime)
python src/compute_cross_fold_stats.py --model cbam_resnet50 --use_metrics


# ========================================
# Utility Commands
# ========================================

# Check GPU
nvidia-smi

# List all checkpoints with timestamps
ls -la outputs/checkpoints/*.pth

# List all metrics JSON files
ls -la outputs/results/*_metrics.json

# View detailed run metrics
cat outputs/results/cbam_resnet50_*_fold0_metrics.json | python -m json.tool

# View training history
cat outputs/logs/cbam_resnet50_*_fold0_history.json | python -m json.tool

# Check specific checkpoint info
python -c "import torch; c = torch.load('outputs/checkpoints/cbam_resnet50_YYYYMMDD_HHMMSS_fold0_best.pth', map_location='cpu'); print(f'Run: {c[\"run_tag\"]}\\nEpoch: {c[\"epoch\"]}\\nKappa: {c[\"best_kappa\"]:.4f}')"

# Compress results with timestamps preserved
tar -czvf phase2-results-$(date +%Y%m%d_%H%M%S).tar.gz outputs/
```

---

## Appendix: Code Verification Checklist

Before running on Vast.ai, verify these files are ready:

| File | Status | Verification Command |
|------|--------|---------------------|
| `src/evaluate.py` | ✅ Ready | `python -c "from evaluate import mc_dropout_inference; print('OK')"` |
| `src/train.py` | ✅ Ready | `python src/train.py --help` (check --model argument exists) |
| `src/model.py` | ✅ Ready | `python -c "from model import create_model, create_baseline_model; print('OK')"` |
| `src/dataset.py` | ✅ Ready | `python -c "from dataset import MessidorDataset; print('OK')"` |
| `src/config.py` | ✅ Ready | `python -c "from config import MESSIDOR_CSV, MESSIDOR_IMAGES; print('OK')"` |
| `src/compute_cross_fold_stats.py` | ⬜ Create | Needs to be created (script provided above) |

---

## Next Steps After Phase 2

After completing Phase 2, proceed to:

1. **Phase 3**: ECE computation and calibration analysis
2. **Phase 4**: Ablation study (4 model variants: baseline, baseline+CA, baseline+SA, CBAM)
3. **Phase 5**: Thesis writing with publication-quality figures
