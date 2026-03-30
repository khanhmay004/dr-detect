# Vast.ai Deployment Guide — Phase 1 GPU Training

> **Purpose**: Train `baseline_resnet50` and `cbam_resnet50` on Vast.ai GPU.
> **Last Updated**: 2026-03-31
> **Interface**: Vast.ai Web Terminal (no SSH needed)
> **Source Code**: GitHub → `git clone`
> **Data**: Google Drive → `gdown`

---

## Prerequisites

Before starting, make sure:

- [x] Phase 0 + Phase 1 CPU smoke tests passed
- [ ] **Latest code is pushed to GitHub** (including the `dataset.py` augmentation fix!)
- [x] Datasets uploaded to Google Drive (`data_dr/` folder, shared)

> **IMPORTANT**: Before renting a Vast.ai instance, push your latest code:
>
> ```powershell
> cd C:\Projects\dr-detect
> git add -A
> git commit -m "Phase 1: fix augmentation API + ready for GPU training"
> git push origin main
> ```

---

## STEP 1: Rent a Vast.ai Instance

1. Go to [https://cloud.vast.ai/](https://cloud.vast.ai/)
2. Filter instances:
   - **GPU**: RTX 3060 / 3070 / 4070 / 3080 (8–12 GB VRAM)
   - **Disk**: ≥ 30 GB
   - **Docker Image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (or latest)
3. Click **RENT**
4. Wait for instance to start (~1–2 minutes)
5. Click the **terminal icon** to open the **Web Terminal**

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
# Expected: config.py  configs/  dataset.py  evaluate.py  loss.py  model.py  preprocessing.py  train.py
```

---

## STEP 3: Download Data from Google Drive

### 3.1 Install gdown

```bash
pip install gdown
```

### 3.2 Share your Google Drive folder

Make sure your `data_dr` folder is shared:

- Right-click `data_dr` → **Share** → **"Anyone with the link"** → **Viewer**
- Copy the folder link

### 3.3 Download datasets

```bash
cd /workspace/dr-detect

# Download the entire data_dr folder from Google Drive
# Replace <FOLDER_ID> with the ID from your Google Drive link
# Example: if link is https://drive.google.com/drive/folders/1aBcDeFgHiJk
# then FOLDER_ID = 1aBcDeFgHiJk
gdown --folder "https://drive.google.com/drive/folders/1lVD77X95Ucpp0npsHUY3AXHyykulkVbP" -O /workspace/dr-detect/data_dr
```

### 3.4 Extract and arrange datasets

Your Drive has `aptos2019-blindness-detection` (zip) and `messidor-2.rar`:

```bash
# Install extraction tools
apt-get update && apt-get install -y unzip unrar

# --- APTOS dataset ---
cd /workspace/dr-detect
# The zip doesn't have a parent folder, so extract it directly into the expected path
mkdir -p aptos/aptos2019-blindness-detection
unzip data_dr/aptos2019-blindness-*.zip -d aptos/aptos2019-blindness-detection/
# If it downloaded as a folder, move it:
# mv data_dr/aptos2019-blindness-detection aptos/

# --- Messidor-2 dataset ---
mkdir -p messidor-2
unrar x data_dr/messidor-2.rar messidor-2/

# --- Cleanup to free disk space ---
rm -rf data_dr/
```

### 3.5 Verify directory structure

```bash
echo "=== APTOS ==="
ls aptos/aptos2019-blindness-detection/
# Expected: train.csv  train_images/  test.csv  ...

echo "=== APTOS image count ==="
ls aptos/aptos2019-blindness-detection/train_images/ | wc -l
# Expected: 3662

echo "=== Messidor-2 ==="
ls messidor-2/
# Expected: IMAGES/  messidor-2.csv  ...

echo "=== Messidor-2 image count ==="
ls messidor-2/IMAGES/ | wc -l
# Expected: 1744 or similar

echo "=== Disk space ==="
df -h /workspace/
```

---

## STEP 4: Install Python Dependencies

> **No `venv` or `conda activate` needed!** On Vast.ai, you're already inside a
> Docker container (`pytorch/pytorch:...`) which acts as the isolated environment.
> Everything you `pip install` goes into the container's Python directly.
> This is different from your local Windows setup where you use `dr-env\Scripts\activate.bat`.

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
GPU:         NVIDIA GeForce RTX 3070
VRAM:        8.0 GB
```

If CUDA is False:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## STEP 6: Train Baseline Model (Phase 1)

### 6.1 Run training

```bash
cd /workspace/dr-detect

# Baseline ResNet-50: 20 epochs, fold 0, batch=16, AMP on
python src/train.py \
    --model baseline \
    --epochs 20 \
    --batch_size 16 \
    --fold 0 \
    2>&1 | tee training_baseline_fold0.log
```

> `tee` saves output to a log file AND prints it to the terminal.

### 6.2 If you need to close the browser tab

Use `nohup` so training continues even if the web terminal disconnects:

```bash
cd /workspace/dr-detect

nohup python src/train.py \
    --model baseline \
    --epochs 20 \
    --batch_size 16 \
    --fold 0 \
    > training_baseline_fold0.log 2>&1 &

# Check it's running
ps aux | grep train.py
```

Reconnect later and check progress:

```bash
tail -30 /workspace/dr-detect/training_baseline_fold0.log
```

### 6.3 Monitor GPU usage

Open a second terminal tab on Vast.ai (if available), or check periodically:

```bash
nvidia-smi
```

### 6.4 Expected timeline

| GPU      | Per Epoch  | 20 Epochs | Cost         |
| -------- | ---------- | --------- | ------------ |
| RTX 3060 | ~12–15 min | ~4–5 hrs  | ~$0.60–$1.50 |
| RTX 3070 | ~8–12 min  | ~3–4 hrs  | ~$0.45–$1.20 |
| RTX 4070 | ~6–8 min   | ~2–3 hrs  | ~$0.30–$0.90 |

### 6.5 Expected final metrics

| Metric       | Expected Range |
| ------------ | -------------- |
| Best Val QWK | 0.75 – 0.85    |
| Best Val AUC | 0.85 – 0.95    |
| Best Val Acc | 0.72 – 0.80    |

---

## STEP 7: Train CBAM Model (Optional — Same Session)

If you have time and budget, train CBAM in the same session:

```bash
cd /workspace/dr-detect

python src/train.py \
    --model cbam \
    --epochs 20 \
    --batch_size 16 \
    --fold 0 \
    2>&1 | tee training_cbam_fold0.log
```

---

## STEP 8: Run Messidor-2 Evaluation

After training completes:

```bash
cd /workspace/dr-detect

python src/evaluate.py \
    --checkpoint outputs/checkpoints/baseline_resnet50_fold0_best.pth \
    --model baseline \
    --batch_size 16 \
    --mc_passes 20
```

---

## STEP 9: Download Results

### Option A: Upload to Google Drive via rclone (Recommended)

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
# Package results first
cd /workspace/dr-detect
tar -czf /workspace/dr-results.tar.gz \
    outputs/ \
    training_baseline_fold0.log \
    training_cbam_fold0.log 2>/dev/null

# Upload to Google Drive
rclone copy /workspace/dr-results.tar.gz gdrive:DR-Detect-Results -P
```

### Option B: Download via SCP (from Windows PowerShell)

If you need to use SCP, get the SSH details from Vast.ai dashboard:

```powershell
scp -P <PORT> -i "C:\Users\ADMIN\.ssh\id_ed25519" ^
    root@<IP>:/workspace/dr-results.tar.gz ^
    "C:\Projects\dr-detect\outputs\vastai-results.tar.gz"
```

### Files you'll get

```
outputs/
├── checkpoints/
│   ├── baseline_resnet50_fold0_best.pth     # Best model (~95 MB)
│   └── baseline_resnet50_fold0_last.pth     # Last epoch checkpoint
├── logs/
│   └── baseline_resnet50_fold0_history.json # Training curves
├── results/                                  # (from evaluate.py)
│   └── messidor2_*.csv / .json
└── figures/                                  # (from evaluate.py)
    └── *.png
```

---

## STEP 10: Destroy the Instance

> **CRITICAL**: Stop billing immediately after downloading results!

On the Vast.ai dashboard → click **Destroy** on your instance.

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size from 16 to 8
python src/train.py --model baseline --epochs 20 --batch_size 8 --fold 0
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
tail -20 training_baseline_fold0.log  # Check latest output
```

### Wrong Directory Structure

If paths don't match, check what `config.py` expects:

```bash
python -c "
from src.config import APTOS_TRAIN_CSV, APTOS_TRAIN_IMAGES, MESSIDOR_CSV, MESSIDOR_IMAGES
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
messidor-2/messidor-2.csv
messidor-2/IMAGES/
```

---

## Quick Reference (Copy-Paste Block)

```bash
# === ON VAST.AI WEB TERMINAL ===

# 1. Clone code
cd /workspace
git clone https://github.com/khanhmay004/dr-detect.git
cd dr-detect

# 2. Get data
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/<YOUR_FOLDER_ID>" -O /workspace/dr-detect/data_dr
apt-get update && apt-get install -y unzip unrar
mkdir -p aptos && unzip data_dr/aptos2019-blindness-*.zip -d aptos/
mkdir -p messidor-2 && unrar x data_dr/messidor-2.rar messidor-2/
rm -rf data_dr/

# 3. Install deps
pip install -r requirements.txt

# 4. Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
ls aptos/aptos2019-blindness-detection/train_images/ | wc -l  # expect 3662

# 5. Train baseline
python src/train.py --model baseline --epochs 20 --batch_size 16 --fold 0 2>&1 | tee training_baseline_fold0.log

# 6. Train CBAM (optional)
python src/train.py --model cbam --epochs 20 --batch_size 16 --fold 0 2>&1 | tee training_cbam_fold0.log

# 7. Evaluate
python src/evaluate.py --checkpoint outputs/checkpoints/baseline_resnet50_fold0_best.pth --model baseline

# 8. Package results
tar -czf /workspace/dr-results.tar.gz outputs/ training_*.log

# 9. Upload (rclone) or download (scp) results → then DESTROY INSTANCE!
```
