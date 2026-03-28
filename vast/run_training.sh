#!/bin/bash
# Training launch script for Vast.ai
# Automatically trains models and saves results

set -e

echo "=========================================="
echo "DR-Detect Training on Vast.ai"
echo "=========================================="
echo ""

# Parse arguments
MODEL="${1:-resnet50}"
EPOCHS="${2:-20}"
FOLD="${3:-0}"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Fold: $FOLD"
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Start training
echo "=========================================="
echo "Starting Training..."
echo "=========================================="
echo ""

cd /workspace/dr-detect

python src/train.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --fold "$FOLD" \
    2>&1 | tee "training_${MODEL}_fold${FOLD}.log"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""

# Check if best checkpoint exists
CHECKPOINT="outputs/checkpoints/${MODEL}_fold${FOLD}_best.pth"
if [ -f "$CHECKPOINT" ]; then
    echo "✓ Best checkpoint saved: $CHECKPOINT"
    
    # Run evaluation
    echo ""
    echo "Running evaluation..."
    python src/evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --model "$MODEL" \
        --fold "$FOLD" \
        2>&1 | tee "evaluation_${MODEL}_fold${FOLD}.log"
    
    echo ""
    echo "=========================================="
    echo "Results Summary"
    echo "=========================================="
    echo ""
    echo "Checkpoint: $CHECKPOINT"
    echo "Metrics: outputs/results/${MODEL}_fold${FOLD}_metrics.json"
    echo "Confusion Matrix: outputs/figures/${MODEL}_fold${FOLD}_confusion_matrix.png"
    echo "ROC Curve: outputs/figures/${MODEL}_fold${FOLD}_roc_curve.png"
    echo ""
    echo "Training Log: training_${MODEL}_fold${FOLD}.log"
    echo "Evaluation Log: evaluation_${MODEL}_fold${FOLD}.log"
    echo ""
    
    # Show metrics if available
    METRICS_FILE="outputs/results/${MODEL}_fold${FOLD}_metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        echo "Key Metrics:"
        python3 -c "import json; d=json.load(open('$METRICS_FILE')); print(f\"  Accuracy: {d['accuracy']:.4f}\"); print(f\"  Kappa: {d['quadratic_kappa']:.4f}\"); print(f\"  AUC: {d['binary_referable']['auc']:.4f}\")"
        echo ""
    fi
else
    echo "✗ Checkpoint not found. Check training logs for errors."
fi

echo "=========================================="
echo "To download results, use:"
echo "  scp -r root@<vast-ip>:/workspace/dr-detect/outputs ."
echo "=========================================="
