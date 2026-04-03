#!/bin/bash
cd /workspace
git clone https://github.com/khanhmay004/dr-detect.git
cd dr-detect
ls src/
# Expected: config.py  configs/  dataset.py  evaluate.py  loss.py  model.py  preprocessing.py  train.py
pip install gdown
cd /workspace/dr-detect
gdown --folder "https://drive.google.com/drive/folders/1lVD77X95Ucpp0npsHUY3AXHyykulkVbP" -O /workspace/dr-detect/data_dr
apt-get update && apt-get install -y unzip unrar
mkdir -p data_split
unrar x data_dr/data_split.rar 

mkdir -p messidor-2
unrar x data_dr/messidor-2.rar messidor-2/

cd /workspace/dr-detect
pip install -r requirements.txt

ls messidor-2/

python -c "
import torch
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA avail:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:         {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM:        {vram:.1f} GB')
"

curl https://rclone.org/install.sh | sudo bash

cd /workspace/dr-detect/messidor-2/IMAGES/
for file in *.JPG; do
  if [ -f \"$file\" ]; then
    mv \"$file\" \"${file%.JPG}.jpg\"
  fi
done
cd /workspace/dr-detect

echo \"DONE SETUP!!\"
# 
echo \"=== Disk space ===\"
df -h /workspace/
