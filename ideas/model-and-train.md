# model.py & train.py — Reference

> **Files**: `src/model.py`, `src/train.py`
> **Purpose**: Model architecture definition + end-to-end GPU training pipeline.

---

## model.py

### Architecture Overview

Two model variants sharing the same backbone and head:

```
Input (B, 3, 512, 512)
  └─ ResNet-50 stem  (conv1 → bn1 → relu → maxpool)
  ├─ layer1 → [CBAM(256)]   ← CBAM variant only
  ├─ layer2 → [CBAM(512)]
  ├─ layer3 → [CBAM(1024)]
  ├─ layer4 → [CBAM(2048)]
  └─ AdaptiveAvgPool2d → MCDropout(p=0.5) → Linear(2048, 5)
```

| Class | Params | Notes |
|-------|--------|-------|
| `CBAMResNet50` | ~25.6M | CBAM after every ResNet stage |
| `BaselineResNet50` | ~23.5M | Identical head, no attention |

### CBAM Components

```
CBAM(C) = ChannelAttention(C) → SpatialAttention()
```

- **ChannelAttention**: avg-pool + max-pool → shared MLP (C → C/16 → C) → sigmoid gate
- **SpatialAttention**: channel avg + channel max → concat → Conv2d(2,1,7×7) → sigmoid gate
- Placed *after* each residual stage — preserves pretrained weights

### MCDropout

```python
class MCDropout(nn.Module):
    def forward(self, x):
        return F.dropout(x, self.p, training=True)  # always-on
```

- Hard-codes `training=True` → dropout fires even in `model.eval()`
- Used for **Monte-Carlo uncertainty estimation** at inference (T=20 passes)
- `mc_active` flag + context manager `model.deterministic_mode()` disables it during validation

### Factory Functions

```python
# CBAM model
model = create_model(num_classes=5, dropout_rate=0.5, pretrained=True)

# Baseline (no attention)
model = create_baseline_model(num_classes=5, dropout_rate=0.5, pretrained=True)
```

### deterministic_mode Context Manager

```python
with model.deterministic_mode():
    logits = model(images)   # dropout disabled → reproducible val metrics
```

Used by `Trainer.validate()` to get stable metrics during training, while MC inference at test time keeps dropout active.

---

## train.py

### CLI

```bash
python src/train.py [OPTIONS]

--model    {baseline, cbam}   # architecture (default: cbam)
--epochs   INT                # default: 20
--lr       FLOAT              # default: 1e-4
--batch_size INT              # default: 16
--fold     INT (0-4)          # which fold to use as validation
--resume   PATH               # resume from checkpoint
--config   YAML_PATH          # optional YAML override (configs/)
--use_cache                   # load from data/processed/ Ben-Graham cache
```

### Training Pipeline (`main()`)

```
seed_everything(42)
  → load APTOS CSV → 5-fold split → DataLoaders
  → build model (baseline | cbam)
  → compute class weights → FocalLoss
  → AdamW optimizer + CosineAnnealingLR
  → Trainer.fit()
```

### Trainer Class

| Method | Description |
|--------|-------------|
| `train_epoch()` | AMP forward/backward + gradient clipping (norm=1.0) |
| `validate()` | deterministic_mode inference → QWK, AUC, Sens, Spec |
| `save_checkpoint()` | saves `_last.pth` every epoch; `_best.pth` on new best QWK |
| `load_checkpoint()` | full resume: model + optimizer + scheduler + scaler state |
| `fit()` | main loop with early stopping (patience=5) |
| `_save_history()` | writes `*_history.json` (per-epoch curves) |
| `_save_run_metrics()` | writes `*_metrics.json` (full run metadata) |

### AMP Training Step

```python
with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
    logits = model(images)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### Validation Metrics

Computed each epoch with `deterministic_mode()` active:

| Metric | How |
|--------|-----|
| **QWK** | `cohen_kappa_score(..., weights="quadratic")` — primary monitor |
| **Accuracy** | `accuracy_score()` |
| **Referable AUC** | `roc_auc_score()` on binary (grade ≥ 2 vs < 2) |
| **Sensitivity** | TP / (TP + FN) from binary confusion matrix |
| **Specificity** | TN / (TN + FP) from binary confusion matrix |

Best model selected by **highest QWK**. Early stop if no improvement for 5 epochs.

### Output Files

All outputs are timestamped: `{model}_{YYYYMMDD_HHMMSS}_fold{N}_*`

```
outputs/
├── checkpoints/
│   ├── *_best.pth     # best QWK checkpoint (~95 MB baseline / ~290 MB CBAM)
│   └── *_last.pth     # most recent epoch
├── logs/
│   └── *_history.json # per-epoch: train_loss, val_kappa, val_auc, ...
└── results/
    └── *_metrics.json # run metadata, best/final metrics, hyperparams, timing
```

### Loss: FocalLoss

```python
criterion = FocalLoss(gamma=2.0, alpha=class_weights)
```

- `alpha` computed from training set class frequencies (inverse-frequency weighting)
- `gamma=2.0` down-weights easy examples, focuses learning on hard/rare classes
- Addresses severe APTOS class imbalance (class 0: 49%, class 4: 6%)

### Optimizer & Scheduler

```python
optimizer = AdamW(params, lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

Cosine schedule decays LR smoothly from `lr` → 0 over `epochs` steps.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CBAM after stage (not inside residual) | Preserves ImageNet pretrained weights |
| `MCDropout` always-on | Enables Bayesian uncertainty without separate inference code |
| `deterministic_mode()` during val | Stable, reproducible val metrics while MC inference is still trivial to activate |
| `GradScaler` + `autocast` | Halves VRAM at 512×512 → enables batch_size=16 on 12 GB GPU |
| `grad_clip_norm=1.0` | Prevents gradient explosion with focal loss on imbalanced data |
| Timestamped checkpoints | Avoids overwriting across multiple runs; `compute_cross_fold_stats.py` picks newest per fold |
| QWK as monitor metric | Standard for ordinal DR grading tasks (Kaggle APTOS competition metric) |

---

## Import Note

All `src/` files use bare imports (`from config import ...`, `from model import ...`).
This works when Python adds `src/` to `sys.path` automatically (i.e. running via `python src/train.py`).
For one-liner `-c` checks, use:

```bash
PYTHONPATH=src python -c "from model import create_model; print('OK')"
```
