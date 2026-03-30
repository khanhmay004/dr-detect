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
   \
   Bảng biến số cần chuẩn bị, cop từ VAST.AI

5. **IP:** (Ví dụ: `154.57.34.91`)

6. **PORT:** (Ví dụ: `19627`)

7. **SSH KEY:** `C:\Users\ADMIN\.ssh\id_ed25519` (Đường dẫn key trên máy)
   ![[Pasted image 20251213194325.png]]
   ![[Pasted image 20251213194303.png]]

---

apt-get update && apt-get install -y zip unzip unrar p7zip-full

# Print the public key

Click dis to create ssh or whaater
[SSH Connection - Vast.ai Documentation – Affordable GPU Cloud Marketplace](https://docs.vast.ai/documentation/instances/connect/ssh#terminal)

```powershell
cat ~/.ssh/id_ed25519.pub
ssh-ed25519 AAAAC3NzaC1lZ9DdI1NTE5AAAAIHWGYlMT8CxcILI/i3DsRvX74HNChkm4JSNFu0wmcv0a
```

# PHẦN 1: GOOGLE DRIVE <-> VAST.AI (Tốc độ cao nhất)

## 1. Chiều UP (Google Drive -> Vast.ai)

Dùng **`gdown`**

- **Chuẩn bị:** Vào Google Drive -> Chuột phải folder dữ liệu -> Share -> **"Anyone with the link"**.

- **Thực hiện (Trên Terminal Vast.ai):**

```bash

# 1. Cài đặt (Mỗi lần thuê máy mới phải chạy lại dòng này)

pip install gdown

# 2. Tải folder (Thay LINK_DRIVE và TÊN_FOLDER)

gdown --folder "LINK_GOOGLE_DRIVE_CUA_BAN" -O /workspace/ten_folder_data

```

## 2. Chiều DOWN (Vast.ai -> Google Drive)

Dùng **`rclone`**. Bắt buộc phải cấu hình vì `gdown` không upload được.
![[Pasted image 20251213194046.png]]

- Bước 0: Set up rclone trên máy
- Vo trong foulder co chua rclone.exe + shift + chuot phai mo powershelll
  rclone authorize "drive" "eyJzY29wZSI6ImRyaXZlIn0" -

```powershell
.\rclone.exe authorize "drive" "eyJzY29wZSI6ImRyaXZlIn0"
```

- **Bước 1: Cài và kết nối (Trên Terminal Vast.ai)**

```bash
# Cài đặt
curl https://rclone.org/install.sh | sudo bash
# Cấu hình (Chỉ cần làm 1 lần mỗi khi thuê máy mới)
rclone config
```

- Nhập `n` (New) -> Tên: `gdrive` -> Chọn số `22 (Google Drive).

- Enter liên tục để bỏ qua Client ID/Secret.

- Scope: Chọn `1` (Full access).

- Enter bỏ qua root folder/service account.

- **Edit advanced config:** `n` (No).

- **Use auto config:** **`n` (No)** (Quan trọng!).

- **Copy dòng lệnh** `rclone authorize "drive"` -> Dán vào Terminal máy tính Windows của bạn -> Đăng nhập trình duyệt -> Copy mã code -> Dán lại vào Vast.ai.
  \*![[Pasted image 20251213193653.png]]

- **Bước 2: Upload dữ liệu (Trên Terminal Vast.ai)**

```powershell
# Cú pháp: rclone copy <NGUỒN_VAST> <ĐÍCH_DRIVE> -P
rclone copy /workspace/nlp/ket_qua gdrive:Backup_Vast -P
```

rclone copy workspace/final_nlp/output.zip gdrive:Output_CsGo -P
zip -r output.zip /workspace/final_nlp/training/outputs
_(Cờ `-P` để hiện thanh phần trăm tiến độ)_

---

# WINDOWS <-> VAST.AI (Trực tiếp)

_Dùng khi dữ liệu nằm sẵn trong máy tính hoặc mạng nhà bạn đủ mạnh._

_Lưu ý: Chạy lệnh trên **PowerShell** của Windows._

## 1. Chiều UP (Windows -> Vast.ai)

Dùng lệnh `scp` (Copy qua SSH).

```powershell

# Cú pháp mẫu

scp -P <PORT> -i "<ĐƯỜNG_DẪN_KEY>" -r "<FOLDER_MÁY_TÍNH>" root@<IP>:/workspace/<TÊN_FOLDER_MỚI>

```

**Ví dụ thực tế (Copy dán và thay số):**

```powershell

scp -P 19627 -i "C:\Users\ADMIN\.ssh\id_ed25519" -r "D:\Datasets\nlp_data" root@154.57.34.91:/workspace/nlp_data

```

## 2. Chiều DOWN (Vast.ai -> Windows)Đảo ngược vị trí nguồn và đích của lệnh trên.

```powershell

# Cú pháp mẫu

scp -P <PORT> -i "<ĐƯỜNG_DẪN_KEY>" -r root@<IP>:/workspace/<FOLDER_VAST> "<ĐƯỜNG_DẪN_MÁY_TÍNH>"

```

**Ví dụ thực tế (Copy dán và thay số):**

```powershell

scp -P 19627 -i "C:\Users\ADMIN\.ssh\id_ed25519" -r root@154.57.34.91:/workspace/nlp/ket_qua "C:\Users\ADMIN\Downloads\KetQua_Model"

```

![[Pasted image 20251213194205.png]]

```
apt-get update && apt-get install -y zip unzip unrar p7zip-full
```

### 2. Hướng dẫn chi tiết từng loại file

````
    # Giải nén tại chỗ
    unzip file_du_lieu.zip

    # Giải nén vào thư mục cụ thể (Ví dụ vào folder /workspace/data)
    unzip file_du_lieu.zip -d /workspace/data
    # Cú pháp: zip -r <tên_file_tạo_ra.zip> <folder_muốn_nén>
    zip -r ket_qua.zip /workspace/nlp/output_folder
    ```



#### B. File `.rar`

- **Giải nén (Unrar):**

    Bash

    ```
    # Giải nén giữ nguyên cấu trúc thư mục
    unrar x file_du_lieu.rar

    # Giải nén vào đường dẫn cụ thể (Lưu ý không có dấu cách sau folder đích)
    unrar x file_du_lieu.rar /workspace/data/
    ```


#### C. File `.tar.gz` hoặc `.tgz` (Thường gặp trong dataset Linux)

- **Giải nén (Tar):**

    Bash

    ```
    # x: extract, z: gzip, v: verbose (hiện tên file), f: file
    tar -xzvf file_du_lieu.tar.gz

    # Giải nén vào thư mục khác (dùng tham số -C)
    tar -xzvf file_du_lieu.tar.gz -C /workspace/data
    ```


#### D. File `.7z` (Hoặc file nén nào `unrar` không mở được)

- **Giải nén (7zip):**

    Bash

    ```
    # x: extract với đường dẫn đầy đủ
    7z x file_du_lieu.7z
    ```

````

# Kiểm tra dung lượng còn trống

```
df -h /workspace/

- **Size:** Tổng dung lượng.

- **Avail:** Dung lượng còn trống (Quan trọng nhất).


Nếu giải nén xong mà hết chỗ, nhớ xóa file nén gốc đi:

Bash

rm file_du_lieu.zip
```
