#!/bin/bash
# Setup script for Vast.ai instance
# Usage: bash setup_vastai.sh <data_download_url> <src_download_url>

set -e  # Exit on error

echo "=========================================="
echo "DR-Detect Vast.ai Setup"
echo "=========================================="
echo ""

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: bash setup_vastai.sh <data_download_url> <src_download_url>"
    echo ""
    echo "Example:"
    echo "  bash setup_vastai.sh https://drive.google.com/XXX/aptos-data.tar.gz https://drive.google.com/XXX/dr-detect-src.tar.gz"
    exit 1
fi

DATA_URL=$1
SRC_URL=$2

echo "Data URL: $DATA_URL"
echo "Source URL: $SRC_URL"
echo ""

# Update system
echo "[1/6] Updating system packages..."
apt-get update -qq
apt-get install -y -qq wget curl unzip > /dev/null
echo "✓ System updated"
echo ""

# Create workspace
echo "[2/6] Creating workspace..."
mkdir -p /workspace/dr-detect
cd /workspace/dr-detect
echo "✓ Workspace created at /workspace/dr-detect"
echo ""

# Download source code
echo "[3/6] Downloading source code..."
wget -q --show-progress -O dr-detect-src.tar.gz "$SRC_URL"
echo "✓ Source code downloaded"
echo ""

# Download data
echo "[4/6] Downloading APTOS dataset..."
echo "  This may take 5-10 minutes depending on data size..."
wget -q --show-progress -O aptos-data.tar.gz "$DATA_URL"
echo "✓ Dataset downloaded"
echo ""

# Extract files
echo "[5/6] Extracting files..."
tar -xzf dr-detect-src.tar.gz
tar -xzf aptos-data.tar.gz -C aptos/
echo "✓ Files extracted"
echo ""

# Setup Python environment
echo "[6/6] Setting up Python environment..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Python environment ready"
echo ""

# Verify GPU
echo "=========================================="
echo "Verifying GPU availability..."
echo "=========================================="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Create output directories
python3 -c "from src.config import setup_directories; setup_directories(); print('✓ Output directories created')"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  bash run_training.sh"
echo ""
echo "Or manually:"
echo "  python src/train.py --model resnet50 --epochs 20 --fold 0"
echo ""
echo "=========================================="
