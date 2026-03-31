# Vast.ai Deployment Guide — Phase 3 Rerun (Calibration + Referral Analysis)

> **Purpose**: Re-run Phase 3 evaluation on Vast.ai using existing baseline/CBAM checkpoints to generate calibrated metrics and referral-policy outputs.
> **Last Updated**: 2026-03-31
> **Scope**: Phase 3D rerun + optional Phase 3E (CBAM folds 1–4) from `plans/05-phase3.md`.

---

## Prerequisites

Before renting an instance:

- [x] Phase 3 A–C code is implemented in `src/evaluate.py`
- [ ] Latest code pushed to GitHub
- [ ] Datasets available (`aptos/...`, `messidor-2/...`)
- [ ] Baseline fold-0 best checkpoint available
- [ ] CBAM fold-0 best checkpoint available

Push latest code:

```powershell
cd C:\Projects\dr-detect
git add -A
git commit -m "Phase 3: add ECE/Brier/reliability/referral analysis"
git push origin main
```

---

## STEP 1: Rent Vast.ai Instance

Recommended filters:

- GPU: RTX 3090 / RTX 4090
- Disk: >= 60 GB
- Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

Open Web Terminal after renting.

---

## STEP 2: Clone Repository

```bash
cd /workspace
git clone https://github.com/khanhmay004/dr-detect.git
cd dr-detect
```

---

## STEP 3: Download and Arrange Data

Install tools:

```bash
pip install gdown
apt-get update && apt-get install -y unzip unrar
```

Download data folder from Google Drive:

```bash
cd /workspace/dr-detect
gdown --folder "https://drive.google.com/drive/folders/1lVD77X95Ucpp0npsHUY3AXHyykulkVbP" -O /workspace/dr-detect/data_dr
```

Extract:

```bash
mkdir -p aptos/aptos2019-blindness-detection
unzip data_dr/aptos2019-blindness-*.zip -d aptos/aptos2019-blindness-detection/

mkdir -p messidor-2
unrar x data_dr/messidor-2.rar messidor-2/

rm -rf data_dr/
```

Verify:

```bash
ls aptos/aptos2019-blindness-detection/train_images | wc -l
head -3 messidor-2/messidor_data.csv
ls messidor-2/IMAGES | wc -l
```

---

## STEP 4: Install Dependencies and Verify GPU

```bash
cd /workspace/dr-detect
pip install -r requirements.txt
```

```bash
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
```

---

## STEP 5: Validate Phase 3 CLI and Paths

Check new evaluation arg:

```bash
python src/evaluate.py --help | grep ece_bins
```

Check dataset paths:

```bash
PYTHONPATH=src python -c "
from config import APTOS_TRAIN_CSV, MESSIDOR_CSV, MESSIDOR_IMAGES
print(APTOS_TRAIN_CSV, APTOS_TRAIN_CSV.exists())
print(MESSIDOR_CSV, MESSIDOR_CSV.exists())
print(MESSIDOR_IMAGES, MESSIDOR_IMAGES.exists())
"
```

---

## STEP 6: Ensure Checkpoints Exist

If checkpoints are already on instance:

```bash
ls -lt outputs/checkpoints/baseline_resnet50*fold0_best.pth | head -1
ls -lt outputs/checkpoints/cbam_resnet50*fold0_best.pth | head -1
```

If missing, upload from local or Drive.

---

## STEP 7: Smoke Test (Phase 3 Rerun)

### 7.1 Baseline smoke run

```bash
cd /workspace/dr-detect
python src/evaluate.py \
  --checkpoint "outputs/checkpoints/baseline_resnet50*fold0_best.pth" \
  --model baseline \
  --batch_size 4 \
  --mc_passes 3 \
  --ece_bins 15 \
  --max_images 10
```

### 7.2 CBAM smoke run

```bash
python src/evaluate.py \
  --checkpoint "outputs/checkpoints/cbam_resnet50*fold0_best.pth" \
  --model cbam \
  --batch_size 4 \
  --mc_passes 3 \
  --ece_bins 15 \
  --max_images 10
```

Smoke success criteria:

- No errors
- Metrics JSON includes `ece` and `brier_score`
- New files exist:
  - `*_reliability.png`
  - `*_referral_curve.csv`
  - `*_referral_curve.png`

---

## STEP 8: Full Phase 3 Rerun (Baseline + CBAM)

### 8.1 Baseline full rerun

```bash
cd /workspace/dr-detect
python src/evaluate.py \
  --checkpoint "outputs/checkpoints/baseline_resnet50*fold0_best.pth" \
  --model baseline \
  --batch_size 16 \
  --mc_passes 20 \
  --ece_bins 15 \
  2>&1 | tee phase3_eval_baseline_messidor2.log
```

### 8.2 CBAM full rerun

```bash
python src/evaluate.py \
  --checkpoint "outputs/checkpoints/cbam_resnet50*fold0_best.pth" \
  --model cbam \
  --batch_size 16 \
  --mc_passes 20 \
  --ece_bins 15 \
  2>&1 | tee phase3_eval_cbam_messidor2.log
```

Expected new outputs per model:

```text
outputs/results/{model}_messidor2_*_20T_1744img_metrics.json
outputs/results/{model}_messidor2_*_20T_1744img_uncertainty.csv
outputs/results/{model}_messidor2_*_20T_1744img_referral_curve.csv
outputs/figures/{model}_messidor2_*_20T_1744img_entropy_hist.png
outputs/figures/{model}_messidor2_*_20T_1744img_conf_vs_ent.png
outputs/figures/{model}_messidor2_*_20T_1744img_reliability.png
outputs/figures/{model}_messidor2_*_20T_1744img_referral_curve.png
```

---

## STEP 9: Compare Baseline vs CBAM (Calibrated Metrics)

Quick comparison helper:

```bash
python - << 'PY'
import json, glob

def latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

b = latest("outputs/results/baseline_messidor2_*_metrics.json")
c = latest("outputs/results/cbam_messidor2_*_metrics.json")
print("Baseline metrics:", b)
print("CBAM metrics:    ", c)

for name, path in [("Baseline", b), ("CBAM", c)]:
    data = json.load(open(path))
    print(f"\n{name}")
    print("  accuracy:", data["accuracy"])
    print("  qwk:", data["quadratic_kappa"])
    print("  referable_auc:", data["binary_referable_auc"])
    print("  referable_sens:", data["binary_referable_sens"])
    print("  referable_spec:", data["binary_referable_spec"])
    print("  ece:", data["ece"])
    print("  brier_score:", data["brier_score"])
PY
```

---

## STEP 10 (Optional): Phase 3E CBAM folds 1–4

If you want robust CBAM stats now:

```bash
cd /workspace/dr-detect
for fold in 1 2 3 4; do
  python src/train.py --model cbam --epochs 20 --batch_size 16 --fold $fold \
    2>&1 | tee "phase3_training_cbam_fold${fold}.log"
done
```

Aggregate:

```bash
python src/compute_cross_fold_stats.py --model cbam_resnet50 --use_metrics \
  2>&1 | tee phase3_cbam_crossfold.log
```

---

## STEP 11: Package and Download Phase 3 Results

```bash
cd /workspace/dr-detect
tar -czf /workspace/phase3-results.tar.gz \
  outputs/results \
  outputs/figures \
  phase3_eval_baseline_messidor2.log \
  phase3_eval_cbam_messidor2.log \
  phase3_cbam_crossfold.log \
  phase3_training_cbam_fold*.log 2>/dev/null
```

Download via SCP (Windows PowerShell):

```powershell
scp -P <PORT> -i "C:\Users\ADMIN\.ssh\id_ed25519" `
  root@<IP>:/workspace/phase3-results.tar.gz `
  "C:\Projects\dr-detect\outputs\phase3-results.tar.gz"
```

---

## STEP 12: Destroy Instance

Destroy instance immediately after download to stop billing.

---

## Troubleshooting

OOM:

```bash
# Use smaller batch size
python src/evaluate.py --checkpoint "..." --model baseline --batch_size 8 --mc_passes 20
```

No checkpoint matched:

```bash
ls -lt outputs/checkpoints/*fold0_best.pth | head
```

Wrong dataset path:

```bash
PYTHONPATH=src python -c "from config import MESSIDOR_CSV; print(MESSIDOR_CSV)"
```

