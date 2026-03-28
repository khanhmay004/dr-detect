# Vast.ai Deployment Guide - DR Detection

Hướng dẫn đầy đủ để deploy và train model trên Vast.ai với GPU.

---

## 📋 Preparation Steps (On Local Machine)

### 1. Compress Data for Upload

```bash
# Activate environment first
dr-env\Scripts\activate.bat

# Run preparation script
python prepare_upload.py
```

Script này sẽ tạo:

- `dr-detect-src.tar.gz` - Source code (~5-10 MB)
- `aptos-data.tar.gz` - APTOS dataset (~1-2 GB)
- `upload_manifest.json` - File checksums và instructions

### 2. Upload to Cloud Storage

**Option A: Google Drive** (Recommended)

1. Upload cả 2 file `.tar.gz` lên Google Drive
2. Share files với quyền "Anyone with the link can view"
3. Lấy direct download links:
   - Vào https://sites.google.com/site/gdocs2direct/
   - Paste Google Drive links
   - Copy direct download links

**Option B: Kaggle Datasets**

1. Upload lên Kaggle Datasets
2. Make dataset public
3. Use download API links

**Option C: Dropbox/OneDrive**

- Similar process, get direct download links

### 3. Note Down URLs

Bạn sẽ cần 2 URLs:

```
DATA_URL=https://...../aptos-data.tar.gz
SRC_URL=https://...../dr-detect-src.tar.gz
```

---

## 🚀 Vast.ai Instance Setup

### 1. Rent GPU Instance

1. Vào https://vast.ai/
2. Search for instance:
   - **GPU**: RTX 3060/3070/4070 (8-12 GB VRAM đủ)
   - **Disk Space**: ≥ 20 GB
   - **Template**: PyTorch (hoặc custom PyTorch 2.0+)
   - **Price**: $0.15 - $0.30/hour

3. Launch instance với:
   ```
   Image: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
   ```

### 2. Connect to Instance

Sau khi instance ready, connect qua SSH:

```bash
ssh -p <port> root@<instance-ip>
```

Hoặc dùng Vast.ai web terminal.

### 3. Run Setup Script

```bash
# Download setup script
wget https://raw.githubusercontent.com/YOUR_REPO/setup_vastai.sh

# Make executable
chmod +x setup_vastai.sh

# Run setup with your URLs
bash setup_vastai.sh <DATA_URL> <SRC_URL>
```

**Example:**

```bash
bash setup_vastai.sh \
  "https://drive.google.com/uc?id=XXX&export=download" \
  "https://drive.google.com/uc?id=YYY&export=download"
```

Setup script sẽ:

- ✓ Update system packages
- ✓ Download và extract data
- ✓ Download và extract source code
- ✓ Install Python dependencies
- ✓ Verify GPU availability
- ✓ Create output directories

---

## 🏋️ Training

### Option 1: Use Training Script (Recommended)

```bash
cd /workspace/dr-detect

# Download training script
wget https://raw.githubusercontent.com/YOUR_REPO/run_training.sh
chmod +x run_training.sh

# Run training (ResNet50, 20 epochs, fold 0)
bash run_training.sh resnet50 20 0

# For EfficientNet
bash run_training.sh efficientnet_b0 20 0
```

### Option 2: Manual Training

```bash
cd /workspace/dr-detect

# Train ResNet50
python src/train.py --model resnet50 --epochs 20 --fold 0

# After training, evaluate
python src/evaluate.py \
  --checkpoint outputs/checkpoints/resnet50_fold0_best.pth \
  --model resnet50 \
  --fold 0
```

---

## 📊 Monitoring Training

### View Live Progress

```bash
# Monitor training log
tail -f training_resnet50_fold0.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Expected Timeline on GPU

- **RTX 3060/3070**: ~10-15 minutes per epoch
- **Total for 20 epochs**: ~3-5 hours
- **Cost estimate**: $0.75 - $1.50

---

## 💾 Download Results

### After Training Completes

```bash
# On your local machine, download results:
scp -P <port> -r root@<instance-ip>:/workspace/dr-detect/outputs ./vastai_results

# Or download specific files:
scp -P <port> root@<instance-ip>:/workspace/dr-detect/outputs/checkpoints/resnet50_fold0_best.pth ./
scp -P <port> root@<instance-ip>:/workspace/dr-detect/outputs/results/resnet50_fold0_metrics.json ./
```

### Files to Download

```
outputs/
├── checkpoints/
│   └── resnet50_fold0_best.pth         # Trained model
├── results/
│   └── resnet50_fold0_metrics.json     # Metrics
├── figures/
│   ├── resnet50_fold0_confusion_matrix.png
│   └── resnet50_fold0_roc_curve.png
└── logs/
    └── resnet50_fold0_history.json     # Training history
```

---

## 🔧 Troubleshooting

### Out of Memory Error

Reduce batch size in `src/config.py`:

```python
BATCH_SIZE = 8  # Instead of 16
```

### Slow Download Speed

Upload data to a faster CDN:

- Use Kaggle Datasets API
- Use AWS S3 with public access
- Use transfer.sh for temporary storage

### CUDA Not Available

Verify PyTorch installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If false, reinstall PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 💰 Cost Optimization

### Training trên tất cả 5 folds

```bash
# Run all folds sequentially
for fold in 0 1 2 3 4; do
  bash run_training.sh resnet50 20 $fold
done
```

**Estimated total cost**: ~$5-7 for complete 5-fold CV

### Stop Instance After Training

```bash
# On Vast.ai dashboard, destroy instance to stop billing
# Or use API:
vastai destroy instance <instance-id>
```

---

## 📝 Quick Reference

### Prepare Data (Local)

```bash
python prepare_upload.py
# Upload .tar.gz files to cloud
```

### Setup Vast.ai

```bash
bash setup_vastai.sh <DATA_URL> <SRC_URL>
```

### Train Model

```bash
bash run_training.sh resnet50 20 0
```

### Download Results (Local)

```bash
scp -P <port> -r root@<ip>:/workspace/dr-detect/outputs ./results
```

---

## 🎯 Expected Results

Sau khi training xong, bạn sẽ có:

- **Validation Kappa**: 0.70 - 0.85 (good baseline)
- **Validation Accuracy**: 75% - 85%
- **Binary Referable AUC**: 0.88 - 0.93
- **Training time**: 3-5 hours trên RTX 3060/3070
- **Total cost**: ~$1-2 cho 1 fold

---

## Next Steps After Training

1. Download tất cả results về máy local
2. So sánh metrics giữa các models (ResNet50 vs EfficientNet)
3. Analyze confusion matrix để hiểu lỗi của model
4. Nếu kết quả tốt, có thể train trên 5 folds để ensemble
