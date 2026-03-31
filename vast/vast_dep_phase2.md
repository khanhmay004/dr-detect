# Vast.ai Deployment Guide — Phase 2: CBAM Training + External Validation

> **Purpose**: Evaluate baseline on Messidor-2, train CBAM-ResNet50 (all 5 folds), and compute cross-fold statistics on Vast.ai GPU.
> **Last Updated**: 2026-03-31
> **Interface**: Vast.ai Web Terminal (no SSH needed)
> **Source Code**: GitHub → `git clone`
> **Data**: Google Drive → `gdown`
> **Phase 1 Reference**: See [`VASTAI_DEPLOYMENT.md`](./VASTAI_DEPLOYMENT.md) for base setup instructions.

---

## Prerequisites

Before starting Phase 2, make sure:

- [x] Phase 1 baseline training is **complete** (checkpoint: `baseline_resnet50_fold0_best.pth` exists)
- [ ] Phase 1 results downloaded locally (checkpoint + history JSON)
- [ ] **Phase 2 code pushed to GitHub** (CBAM model, evaluate.py, compute_cross_fold_stats.py)
- [x] Datasets uploaded to Google Drive (`data_dr/` folder, shared)
- [x] Messidor-2 label file ready (`messidor_data.csv` with adjudicated Krause et al. 2018 grades)

> **IMPORTANT**: Push ALL latest code before renting an instance:
>
> ```powershell
> cd C:\Projects\dr-detect
> git add -A
> git commit -m "Phase 2: CBAM model + evaluation pipeline ready for GPU"
> git push origin main
> ```

### Code Files to Verify Before Push

| File | Required For | Verify Command |
|------|-------------|----------------|
| `src/model.py` | CBAM model architecture | `PYTHONPATH=src python -c "from model import create_model; print('OK')"` |
| `src/train.py` | `--model cbam` argument | `python src/train.py --help` |
| `src/evaluate.py` | MC Dropout evaluation | `PYTHONPATH=src python -c "from evaluate import mc_dropout_inference; print('OK')"` |
| `src/dataset.py` | Messidor-2 loading | `PYTHONPATH=src python -c "from dataset import MessidorDataset; print('OK')"` |
| `src/config.py` | `MESSIDOR_CSV`, paths | `PYTHONPATH=src python -c "from config import MESSIDOR_CSV; print(MESSIDOR_CSV)"` |
| `src/compute_cross_fold_stats.py` | Cross-fold aggregation | `python src/compute_cross_fold_stats.py --help` |

---

## STEP 1: Rent a Vast.ai Instance

1. Go to [https://cloud.vast.ai/](https://cloud.vast.ai/)
2. Filter instances:
   - **GPU**: RTX 3090 / RTX 4090 (24 GB VRAM recommended — CBAM is slightly heavier)
   - **Disk**: ≥ 50 GB (need room for 5 fold checkpoints ~290 MB each)
   - **Docker Image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (or latest)
3. Click **RENT**
4. Wait for instance to start (~1–2 minutes)
5. Click the **terminal icon** to open the **Web Terminal**

> **Cost Estimate**: RTX 3090 ~$0.30–0.50/hr × ~15 hours total ≈ **$5–8**

---

## STEP 2: Clone Source Code from GitHub

```bash
cd /workspace
git clone https://github.com/khanhmay004/dr-detect.git
cd dr-detect
```

Verify source code:

```bash
ls src/
# Expected: config.py  configs/  compute_cross_fold_stats.py  dataset.py
#           evaluate.py  loss.py  model.py  preprocessing.py  train.py
```

---

## STEP 3: Download Data from Google Drive

### 3.1 Install gdown

```bash
pip install gdown
```

### 3.2 Download datasets

```bash
cd /workspace/dr-detect

# Download the entire data_dr folder from Google Drive
# Replace <FOLDER_ID> with the ID from your Google Drive link
gdown --folder "https://drive.google.com/drive/folders/1lVD77X95Ucpp0npsHUY3AXHyykulkVbP" -O /workspace/dr-detect/data_dr
```

### 3.3 Extract and arrange datasets

```bash
# Install extraction tools
apt-get update && apt-get install -y unzip unrar

# --- APTOS dataset ---
cd /workspace/dr-detect
mkdir -p aptos/aptos2019-blindness-detection
unzip data_dr/aptos2019-blindness-*.zip -d aptos/aptos2019-blindness-detection/

# --- Messidor-2 dataset ---
mkdir -p messidor-2
unrar x data_dr/messidor-2.rar messidor-2/

# --- Cleanup to free disk space ---
rm -rf data_dr/
```

### 3.4 Verify directory structure

```bash
echo "=== APTOS ==="
ls aptos/aptos2019-blindness-detection/
# Expected: train.csv  train_images/  test.csv  ...

echo "=== APTOS image count ==="
ls aptos/aptos2019-blindness-detection/train_images/ | wc -l
# Expected: 3662

echo "=== Messidor-2 ==="
ls messidor-2/
# Expected: IMAGES/  messidor_data.csv  ...

echo "=== Messidor-2 CSV check ==="
head -3 messidor-2/messidor_data.csv
# Expected columns: image_id, adjudicated_dr_grade, adjudicated_gradable
wc -l messidor-2/messidor_data.csv
# Expected: 1748 rows (header + 1747 images, 1744 gradable)

echo "=== Messidor-2 image count ==="
ls messidor-2/IMAGES/ | wc -l
# Expected: 1744 or similar

echo "=== Disk space ==="
df -h /workspace/
```

---

## STEP 4: Install Python Dependencies

> **No `venv` or `conda activate` needed!** On Vast.ai, you're already inside a
> Docker container which acts as the isolated environment.

```bash
cd /workspace/dr-detect
pip install -r requirements.txt
```

> `torch` and `torchvision` are pre-installed in the PyTorch Docker image.
> This just adds `albumentations`, `scikit-learn`, `tqdm`, `pyyaml`, etc.

---

## STEP 5: Verify GPU Setup

```bash
python -c "
import torch
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA avail:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:         {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM:        {vram:.1f} GB')
"
```

Expected:

```
PyTorch:     2.x.x
CUDA avail:  True
GPU:         NVIDIA GeForce RTX 3090
VRAM:        24.6 GB
```

If CUDA is False:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## STEP 6: Verify Configuration & Paths

Before running any experiment, sanity-check that all paths resolve correctly:

```bash
cd /workspace/dr-detect

PYTHONPATH=src python -c "
from config import APTOS_TRAIN_CSV, APTOS_TRAIN_IMAGES, MESSIDOR_CSV, MESSIDOR_IMAGES
print(f'APTOS CSV:    {APTOS_TRAIN_CSV}  exists={APTOS_TRAIN_CSV.exists()}')
print(f'APTOS images: {APTOS_TRAIN_IMAGES}  exists={APTOS_TRAIN_IMAGES.exists()}')
print(f'Messidor CSV: {MESSIDOR_CSV}  exists={MESSIDOR_CSV.exists()}')
print(f'Messidor img: {MESSIDOR_IMAGES}  exists={MESSIDOR_IMAGES.exists()}')
"
```

All four should show `exists=True`. If not, fix paths before proceeding.

---

## STEP 7: Phase 2.1 — Messidor-2 Baseline Evaluation

> **Goal**: Evaluate the Phase 1 baseline checkpoint on external data to measure domain generalization.
> **Time**: ~5 minutes

### 7.1 Upload or verify baseline checkpoint

If you trained baseline in a **previous** Vast.ai session, you need to upload the checkpoint:

**Option A** — Upload from Google Drive (if you saved results there):

```bash
# If baseline checkpoint is on Google Drive
gdown "<FILE_ID>" -O outputs/checkpoints/baseline_resnet50_fold0_best.pth
```

**Option B** — Upload via SCP from local Windows machine:

```powershell
# From Windows PowerShell
scp -P <PORT> -i "C:\Users\ADMIN\.ssh\id_ed25519" ^
    "C:\Projects\dr-detect\dr-results\outputs\checkpoints\baseline_resnet50_fold0_best.pth" ^
    root@<IP>:/workspace/dr-detect/outputs/checkpoints/
```

**Option C** — If baseline was trained on **this** same instance, checkpoint already exists:

```bash
ls -la outputs/checkpoints/baseline_resnet50_*_fold0_best.pth
```

### 7.2 Smoke test (verify everything works)

```bash
cd /workspace/dr-detect

# Quick test: 10 images, 3 MC passes (wildcard auto-selects most recent)
python src/evaluate.py \
    --checkpoint "outputs/checkpoints/baseline_resnet50*fold0_best.pth" \
    --model baseline \
    --batch_size 4 \
    --mc_passes 3 \
    --max_images 10
```

Expected: Should complete in ~30 seconds with no errors. You'll see "Resolved checkpoint pattern to: ..." showing which checkpoint was selected.

### 7.3 Full Messidor-2 evaluation

```bash
cd /workspace/dr-detect

python src/evaluate.py \
    --checkpoint "outputs/checkpoints/baseline_resnet50*fold0_best.pth" \
    --model baseline \
    --batch_size 16 \
    --mc_passes 20 \
    2>&1 | tee eval_baseline_messidor2.log
```

### 7.4 Expected output

```
Device: cuda
Loading Messidor-2 dataset ...
  Images: 1744
  (filtered to gradable images only)

Loading BASELINE model from checkpoint ...
  Checkpoint epoch: 14  |  best kappa: 0.9088

Running MC Dropout inference (T = 20) ...
MC Inference (T=20): 100%|██████████| 109/109 [01:30<00:00]

==================================================
  MESSIDOR-2 EVALUATION
==================================================
  Accuracy:          0.XXXX
  Quadratic kappa:   0.XXXX    # Expect 0.55-0.75 (domain shift)
  Referable DR AUC:  0.XXXX    # Expect 0.80-0.92

==================================================
  UNCERTAINTY SUMMARY
==================================================
  Mean entropy:   0.XXXX
  Median entropy: 0.XXXX
  Max entropy:    X.XXXX
  Mean confidence: 0.XXXX
```

### 7.5 Verify outputs

```bash
# Check result files were created
ls -la outputs/results/baseline_messidor2_*
ls -la outputs/figures/baseline_messidor2_*
```

Expected files:

```
outputs/results/baseline_messidor2_*_20T_*_uncertainty.csv
outputs/results/baseline_messidor2_*_20T_*_metrics.json
outputs/figures/baseline_messidor2_*_entropy_hist.png
outputs/figures/baseline_messidor2_*_conf_vs_ent.png
```

### 7.6 Expected metrics (with domain shift)

| Metric | APTOS Val (Fold 0) | Expected Messidor-2 | Notes |
|--------|-------------------|---------------------|-------|
| **QWK** | 0.9088 | 0.55 – 0.75 | Domain shift expected |
| **Accuracy** | 84.45% | 60% – 75% | Different imaging devices |
| **Referable AUC** | 98.46% | 80% – 92% | Binary classification more robust |
| **Mean Entropy** | — | 0.3 – 0.8 | Higher = more uncertain |

---

## STEP 8: Phase 2.2 — CBAM-ResNet50 Fold 0 Training

> **Goal**: Train CBAM with identical hyperparameters as baseline for valid ablation comparison.
> **Time**: ~2.5 hours on RTX 3090

### 8.1 Verify hyperparameters match baseline

The following must be **identical** to baseline training (only the model architecture changes):

| Parameter | Value |
|-----------|-------|
| epochs | 20 |
| batch_size | 16 |
| learning_rate | 1e-4 |
| weight_decay | 1e-4 |
| image_size | 512 |
| focal_gamma | 2.0 |
| dropout_rate | 0.5 |
| grad_clip_norm | 1.0 |
| AMP | True |
| seed | 42 |

### 8.2 Run CBAM fold 0 training

```bash
cd /workspace/dr-detect

python src/train.py \
    --model cbam \
    --epochs 20 \
    --batch_size 16 \
    --fold 0 \
    2>&1 | tee training_cbam_fold0.log
```

> **If you need to close the browser tab**, use `nohup` so training continues:
>
> ```bash
> cd /workspace/dr-detect
>
> nohup python src/train.py \
>     --model cbam \
>     --epochs 20 \
>     --batch_size 16 \
>     --fold 0 \
>     > training_cbam_fold0.log 2>&1 &
>
> # Check it's running
> ps aux | grep train.py
> ```
>
> Reconnect later:
> ```bash
> tail -30 /workspace/dr-detect/training_cbam_fold0.log
> ```

### 8.3 Monitor GPU usage

```bash
nvidia-smi
# Or watch it refresh:
watch -n 5 nvidia-smi
```

### 8.4 Expected training output

```
Device: cuda (NVIDIA GeForce RTX 3090 - 24.6 GB)
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
  ...
```

### 8.5 Verify CBAM fold 0 outputs

```bash
# Check checkpoint
ls -la outputs/checkpoints/cbam_resnet50_*_fold0_best.pth
# Expected: ~290 MB (slightly larger than baseline due to CBAM modules)

# Check training history
ls -la outputs/logs/cbam_resnet50_*_fold0_history.json

# Quick peek at results
cat outputs/logs/cbam_resnet50_*_fold0_history.json | python -m json.tool | tail -20
```

### 8.6 Expected fold 0 metrics

| Metric | Baseline Fold 0 | Expected CBAM Fold 0 |
|--------|----------------|---------------------|
| **Val QWK** | 0.9088 | 0.90 – 0.93 |
| **Val Accuracy** | 84.45% | 83% – 87% |
| **Val AUC** | 98.46% | 98% – 99% |

---

## STEP 9: Phase 2.2b — CBAM Messidor-2 Evaluation

> **Goal**: Evaluate CBAM on Messidor-2 for comparison with baseline external validation.
> **Time**: ~5 minutes

```bash
cd /workspace/dr-detect

# Run evaluation (wildcard auto-selects most recent checkpoint)
python src/evaluate.py \
    --checkpoint "outputs/checkpoints/cbam_resnet50*fold0_best.pth" \
    --model cbam \
    --batch_size 16 \
    --mc_passes 20 \
    2>&1 | tee eval_cbam_messidor2.log
```

Expected: You'll see "Resolved checkpoint pattern to: ..." showing which checkpoint was selected.

### 9.1 Compare baseline vs CBAM on Messidor-2

After both evaluations, quickly compare:

```bash
echo "=== BASELINE Messidor-2 ==="
cat outputs/results/baseline_messidor2_*_metrics.json | python -m json.tool

echo ""
echo "=== CBAM Messidor-2 ==="
cat outputs/results/cbam_messidor2_*_metrics.json | python -m json.tool
```

---

## STEP 10: Phase 2.3 — CBAM Folds 1–4 Training

> **Goal**: Train CBAM on remaining folds for robust cross-validation statistics.
> **Time**: ~10 hours total (4 folds × ~2.5 hours each)
> **This step is optional but recommended** for publication-quality results.

### 10.1 Create batch training script

```bash
cd /workspace/dr-detect

cat > run_cbam_folds.sh << 'EOF'
#!/bin/bash
# run_cbam_folds.sh - Train CBAM on folds 1-4
# (Fold 0 already trained in Step 8)

set -e  # Exit on error

echo "=========================================="
echo "  CBAM-ResNet50 Cross-Validation Training"
echo "  Folds 1-4 (Fold 0 already done)"
echo "=========================================="

for fold in 1 2 3 4; do
    echo ""
    echo ">>> Starting Fold $fold at $(date)"
    echo ""

    python src/train.py \
        --model cbam \
        --epochs 20 \
        --batch_size 16 \
        --fold $fold \
        2>&1 | tee "training_cbam_fold${fold}.log"

    echo ""
    echo ">>> Completed Fold $fold at $(date)"
    echo ""
done

echo "=========================================="
echo "  All folds complete!"
echo "  $(date)"
echo "=========================================="
EOF

chmod +x run_cbam_folds.sh
```

### 10.2 Run folds 1–4

**Option A** — Interactive (keep terminal open):

```bash
cd /workspace/dr-detect
bash run_cbam_folds.sh
```

**Option B** — Background with `nohup` (can close browser):

```bash
cd /workspace/dr-detect

nohup bash run_cbam_folds.sh > training_cbam_all_folds.log 2>&1 &

# Check it's running
ps aux | grep train.py
```

Reconnect later and check:

```bash
tail -50 /workspace/dr-detect/training_cbam_all_folds.log
# Or check individual fold logs:
tail -20 /workspace/dr-detect/training_cbam_fold2.log
```

### 10.3 Expected training timeline

| Fold | Estimated Time | Cumulative |
|------|---------------|------------|
| Fold 1 | ~2.5 hours | 2.5 hours |
| Fold 2 | ~2.5 hours | 5.0 hours |
| Fold 3 | ~2.5 hours | 7.5 hours |
| Fold 4 | ~2.5 hours | 10.0 hours |

### 10.4 Verify all checkpoints

```bash
echo "=== All CBAM checkpoints ==="
ls -la outputs/checkpoints/cbam_resnet50_*_best.pth
# Expected: 5 files (fold0 through fold4), each ~290 MB

echo ""
echo "=== All CBAM histories ==="
ls -la outputs/logs/cbam_resnet50_*_history.json
# Expected: 5 files
```

---

## STEP 11: Phase 2.4 — Cross-Fold Statistics

> **Goal**: Aggregate results across all 5 folds for publication-quality mean ± std.
> **Time**: ~1 minute

### 11.1 Run cross-fold aggregation

```bash
cd /workspace/dr-detect

# Using history files
python src/compute_cross_fold_stats.py --model cbam_resnet50

# Or using detailed metrics JSON (includes runtime info)
python src/compute_cross_fold_stats.py --model cbam_resnet50 --use_metrics
```

### 11.2 Also compute baseline cross-fold stats (if all 5 baseline folds exist)

```bash
python src/compute_cross_fold_stats.py --model baseline_resnet50
```

### 11.3 Expected cross-fold output

```
============================================================
  Cross-Fold Statistics: CBAM_RESNET50
============================================================

  Fold 0: QWK=0.9120, Acc=0.8512, AUC=0.9867
  Fold 1: QWK=0.9045, Acc=0.8423, AUC=0.9845
  Fold 2: QWK=0.9089, Acc=0.8467, AUC=0.9856
  Fold 3: QWK=0.9012, Acc=0.8389, AUC=0.9834
  Fold 4: QWK=0.9078, Acc=0.8456, AUC=0.9851

------------------------------------------------------------
  SUMMARY (5 folds)
------------------------------------------------------------
  Val QWK:      0.9069 +/- 0.0039
  Val Accuracy: 0.8449 +/- 0.0044
  Val AUC:      0.9851 +/- 0.0012
------------------------------------------------------------

  Saved: outputs/results/cbam_resnet50_crossfold_stats_YYYYMMDD_HHMMSS.json
```

---

## STEP 12: Download Results

### 12.1 Package all outputs

```bash
cd /workspace/dr-detect

tar -czf /workspace/phase2-results.tar.gz \
    outputs/ \
    training_cbam_fold*.log \
    eval_baseline_messidor2.log \
    eval_cbam_messidor2.log 2>/dev/null
```

### 12.2 Option A: Upload to Google Drive via rclone (Recommended)

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive
rclone config
# → n (New remote)
# → Name: gdrive
# → Storage: 22 (Google Drive)
# → Enter (skip Client ID)
# → Enter (skip Client Secret)
# → Scope: 1 (Full access)
# → Enter (skip root folder)
# → Enter (skip service account)
# → Advanced config: n
# → Auto config: n (headless server!)
```

It will print a command like `rclone authorize "drive" "eyJ..."`.
**Run that command on your Windows machine** (if you have rclone installed),
or go to the URL it gives you, login, and paste the token back.

Then upload:

```bash
rclone copy /workspace/phase2-results.tar.gz gdrive:DR-Detect-Results -P
```

### 12.3 Option B: Download via SCP (from Windows PowerShell)

```powershell
scp -P <PORT> -i "C:\Users\ADMIN\.ssh\id_ed25519" ^
    root@<IP>:/workspace/phase2-results.tar.gz ^
    "C:\Projects\dr-detect\outputs\phase2-results.tar.gz"
```

### 12.4 Files you'll get

```
outputs/
├── checkpoints/
│   ├── baseline_resnet50_*_fold0_best.pth          # From Phase 1
│   ├── cbam_resnet50_*_fold0_best.pth              # NEW
│   ├── cbam_resnet50_*_fold0_last.pth              # NEW
│   ├── cbam_resnet50_*_fold1_best.pth              # NEW (if folds 1-4 trained)
│   ├── cbam_resnet50_*_fold1_last.pth
│   ├── cbam_resnet50_*_fold2_best.pth
│   ├── cbam_resnet50_*_fold2_last.pth
│   ├── cbam_resnet50_*_fold3_best.pth
│   ├── cbam_resnet50_*_fold3_last.pth
│   ├── cbam_resnet50_*_fold4_best.pth
│   └── cbam_resnet50_*_fold4_last.pth
├── logs/
│   ├── baseline_resnet50_*_fold0_history.json
│   ├── cbam_resnet50_*_fold0_history.json          # NEW
│   ├── cbam_resnet50_*_fold1_history.json          # NEW
│   ├── cbam_resnet50_*_fold2_history.json
│   ├── cbam_resnet50_*_fold3_history.json
│   └── cbam_resnet50_*_fold4_history.json
├── results/
│   ├── baseline_messidor2_*_20T_*_uncertainty.csv  # NEW
│   ├── baseline_messidor2_*_20T_*_metrics.json     # NEW
│   ├── cbam_messidor2_*_20T_*_uncertainty.csv      # NEW
│   ├── cbam_messidor2_*_20T_*_metrics.json         # NEW
│   └── cbam_resnet50_crossfold_stats_*.json        # NEW (cross-fold)
└── figures/
    ├── baseline_messidor2_*_entropy_hist.png        # NEW
    ├── baseline_messidor2_*_conf_vs_ent.png         # NEW
    ├── cbam_messidor2_*_entropy_hist.png            # NEW
    └── cbam_messidor2_*_conf_vs_ent.png             # NEW
```

---

## STEP 13: Destroy the Instance

> **CRITICAL**: Stop billing immediately after downloading results!

On the Vast.ai dashboard → click **Destroy** on your instance.

---

## Troubleshooting

### CUDA Out of Memory

```bash
# CBAM is ~8% heavier than baseline — reduce batch size if OOM:
python src/train.py --model cbam --epochs 20 --batch_size 8 --fold 0
```

### gdown Rate Limit / Fails for Large Files

```bash
# Try with --fuzzy
gdown --fuzzy "https://drive.google.com/file/d/<FILE_ID>/view"

# Or download individual files by file ID
gdown "<FILE_ID>" -O filename.zip
```

### Training Disconnected

If you used `nohup`, training keeps running. Check:

```bash
ps aux | grep train.py          # Is it still running?
tail -20 training_cbam_fold0.log  # Check latest output
```

### Checkpoint Not Found (Timestamped Names)

If checkpoint names have timestamps, use `ls` to find the exact name:

```bash
# List all baseline checkpoints
ls -lt outputs/checkpoints/baseline_resnet50_*_fold0_best.pth | head -1

# List all CBAM checkpoints
ls -lt outputs/checkpoints/cbam_resnet50_*_fold0_best.pth | head -1
```

### Wrong Directory Structure

```bash
PYTHONPATH=src python -c "
from config import APTOS_TRAIN_CSV, APTOS_TRAIN_IMAGES, MESSIDOR_CSV, MESSIDOR_IMAGES
print(f'APTOS CSV:    {APTOS_TRAIN_CSV}  exists={APTOS_TRAIN_CSV.exists()}')
print(f'APTOS images: {APTOS_TRAIN_IMAGES}  exists={APTOS_TRAIN_IMAGES.exists()}')
print(f'Messidor CSV: {MESSIDOR_CSV}  exists={MESSIDOR_CSV.exists()}')
print(f'Messidor img: {MESSIDOR_IMAGES}  exists={MESSIDOR_IMAGES.exists()}')
"
```

Expected paths (relative to project root):

```
aptos/aptos2019-blindness-detection/train.csv
aptos/aptos2019-blindness-detection/train_images/
messidor-2/messidor_data.csv
messidor-2/IMAGES/
```

### Fold Training Fails Mid-Script

If `run_cbam_folds.sh` fails mid-way (e.g., fold 2 crashes), edit the script or run remaining folds manually:

```bash
# Resume from fold 3 onward
for fold in 3 4; do
    python src/train.py --model cbam --epochs 20 --batch_size 16 --fold $fold \
        2>&1 | tee "training_cbam_fold${fold}.log"
done
```

---

## Quick Reference (Copy-Paste Block)

```bash
# ================================================================
# PHASE 2 — COMPLETE WORKFLOW ON VAST.AI WEB TERMINAL
# ================================================================

# --- Setup (same as Phase 1) ---
cd /workspace
git clone https://github.com/khanhmay004/dr-detect.git
cd dr-detect

pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1lVD77X95Ucpp0npsHUY3AXHyykulkVbP" -O /workspace/dr-detect/data_dr
apt-get update && apt-get install -y unzip unrar
mkdir -p aptos/aptos2019-blindness-detection && unzip data_dr/aptos2019-blindness-*.zip -d aptos/aptos2019-blindness-detection/
mkdir -p messidor-2 && unrar x data_dr/messidor-2.rar messidor-2/
rm -rf data_dr/

pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# --- Phase 2.1: Baseline Messidor-2 Evaluation (~5 min) ---
python src/evaluate.py \
    --checkpoint "outputs/checkpoints/baseline_resnet50*fold0_best.pth" \
    --model baseline --batch_size 16 --mc_passes 20 \
    2>&1 | tee eval_baseline_messidor2.log

# --- Phase 2.2: CBAM Fold 0 Training (~2.5 hrs) ---
python src/train.py --model cbam --epochs 20 --batch_size 16 --fold 0 \
    2>&1 | tee training_cbam_fold0.log

# --- Phase 2.2b: CBAM Messidor-2 Evaluation (~5 min) ---
python src/evaluate.py \
    --checkpoint "outputs/checkpoints/cbam_resnet50*fold0_best.pth" \
    --model cbam --batch_size 16 --mc_passes 20 \
    2>&1 | tee eval_cbam_messidor2.log

# --- Phase 2.3: CBAM Folds 1-4 (~10 hrs) ---
for fold in 1 2 3 4; do
    python src/train.py --model cbam --epochs 20 --batch_size 16 --fold $fold \
        2>&1 | tee "training_cbam_fold${fold}.log"
done

# --- Phase 2.4: Cross-fold Statistics ---
python src/compute_cross_fold_stats.py --model cbam_resnet50

# --- Download Results ---
tar -czf /workspace/phase2-results.tar.gz outputs/ training_*.log eval_*.log
# Then upload via rclone or download via scp → DESTROY INSTANCE!
```

---

## Estimated Total Time & Cost

| Phase | Task | Time (RTX 3090) | Cost (~$0.40/hr) |
|-------|------|-----------------|-------------------|
| 2.1 | Baseline Messidor-2 eval | ~5 min | ~$0.03 |
| 2.2 | CBAM fold 0 train | ~2.5 hrs | ~$1.00 |
| 2.2b | CBAM Messidor-2 eval | ~5 min | ~$0.03 |
| 2.3 | CBAM folds 1–4 train | ~10 hrs | ~$4.00 |
| 2.4 | Cross-fold stats | ~1 min | ~$0.01 |
| | Setup + data download | ~15 min | ~$0.10 |
| **Total** | | **~13 hrs** | **~$5–6** |

> **Budget-saving tip**: If tight on budget, skip Phase 2.3 (folds 1–4). Fold 0 alone gives you the CBAM vs baseline ablation comparison and both Messidor-2 evaluations.

---

## Next Steps After Phase 2

After downloading results, proceed to:

1. **Phase 3**: ECE computation and calibration analysis
2. **Phase 4**: Ablation study (4 model variants: baseline, baseline+CA, baseline+SA, CBAM)
3. **Phase 5**: Thesis writing with publication-quality figures
