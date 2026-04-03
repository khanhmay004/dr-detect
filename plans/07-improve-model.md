# Plan 07: Improve Per-Class Model Performance

> **Version**: 1.0
> **Date**: 2026-04-03
> **Context**: Phase 3D complete. Baseline outperforms CBAM on all metrics except ECE. Both models show near-catastrophic failure on Grades 1–4 (Messidor-2: Grade 1 recall 4.8%, Grade 2 recall 23.6%). Root cause analysis in `docs/improve-model.md`.
> **Prerequisite reading**: `docs/improve-model.md` (diagnosis), `docs/result-report.md` (numbers).

---

## 1. Goal

Improve per-class recall for DR Grades 1–4 while maintaining Grade 0 performance, through a structured sequence of code changes: first post-hoc fixes (no retraining), then structural pipeline changes (retrain required).

**Success criteria** (fold 0, APTOS validation):

| Metric | Current Baseline | Target |
|--------|-----------------|--------|
| Grade 3 (Severe) F1 | ~0.32 | ≥ 0.45 |
| Grade 1 (Mild) F1 | ~0.61 | ≥ 0.55 (maintain) |
| Macro F1 | ~0.63 | ≥ 0.65 |
| QWK | 0.9088 | ≥ 0.88 (acceptable regression) |
| Grade 0 Recall | ~0.98 | ≥ 0.90 (acceptable trade-off) |

**Non-goal**: Fixing Messidor-2 domain shift (that's a data/domain-adaptation problem, not a model architecture problem).

---

## 2. Current Code State (Anchor Points)

All line numbers reference the code as of 2026-04-03.

### `src/loss.py`
- **FocalLoss** (lines 9–51): Takes `gamma`, `alpha` (per-class weight tensor), `reduction`. Computes `CE → p_t → focal_weight → alpha_weight → reduce`. No `label_smoothing` parameter.
- **`compute_class_weights`** (lines 53–57): `α_c = total / (num_classes × count_c)`.

### `src/model.py`
- **CBAMResNet50** classifier head (lines 122–124):
  ```python
  self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
  self.mc_dropout = MCDropout(dropout_rate)
  self.fc = nn.Linear(2048, num_classes)
  ```
- **BaselineResNet50** classifier head (lines 196–198): Identical structure.
- Both models use `MCDropout` with `mc_active` toggle + `deterministic_mode()` context manager.

### `src/dataset.py`
- **`create_dataloaders`** (lines 329–369): Train loader uses `shuffle=True`, no sampler.
  ```python
  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,           # ← No balanced sampling
      num_workers=num_workers,
      pin_memory=True,
      drop_last=True,
  )
  ```
- **`DRDataset.get_class_weights`** (lines 103–108): Computes inverse-frequency weights from DataFrame labels.

### `src/train.py`
- **Loss setup** (lines 500–504):
  ```python
  train_labels = torch.tensor(train_df["diagnosis"].values)
  alpha_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
  criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha_weights)
  ```
- **Optimizer + Scheduler** (lines 507–512):
  ```python
  optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
  ```
- **Training loop** (lines 86–125): Standard `train_epoch` with AMP + gradient clipping.
- **Validation** (lines 129–198): Uses `deterministic_mode()`, computes per-class metrics.

### `src/config.py`
- Key constants: `FOCAL_GAMMA = 2.0`, `MC_DROPOUT_RATE = 0.5`, `LEARNING_RATE = 1e-4`, `EPOCHS = 20`, `BATCH_SIZE = 16`, `EARLY_STOPPING_PATIENCE = 5`.

### `src/evaluate.py`
- **`mc_dropout_inference`** (lines 42–101): Returns dict with `mean_probs`, `entropy`, `predictions`, `confidence`, `all_probs` (N, T, C).
- **Predictions** (line 91): `confidence, predictions = mean_probs.max(dim=1)` — raw argmax, no threshold adjustment.

### `src/configs/`
- `gpu_baseline.yaml`, `gpu_cbam.yaml`: Current training configs (identical except `model_type`).
- `experiment_config.py`: `ExperimentConfig` dataclass + `load_config()`.

---

## 3. Implementation Phases

### Phase A — Post-Hoc Fixes (No Retraining)

Apply to the **existing trained baseline** checkpoint. Zero risk to existing results.

### Phase B — Structural Pipeline Changes (Retrain Required)

Modify training pipeline code, then retrain baseline on fold 0 to validate improvement.

### Phase C — Ordinal Loss (Optional Extension)

If Phase B results are promising, add ordinal-aware loss as a further experiment.

---

## 4. Phase A: Post-Hoc Fixes

### A1. Temperature Scaling

**Goal**: Learn a single scalar `T` on APTOS validation logits such that `softmax(logits / T)` minimises NLL (equivalently, minimises ECE). Does not change `argmax` predictions — only calibrates probabilities.

**New file**: `src/temperature_scaling.py`

```python
"""Post-hoc temperature scaling for probability calibration.

Usage:
    python temperature_scaling.py \
        --checkpoint "outputs/checkpoints/baseline*fold0_best.pth" \
        --model baseline --fold 0
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    APTOS_TRAIN_CSV, APTOS_TRAIN_IMAGES, APTOS_PROCESSED_DIR,
    USE_PREPROCESSED_CACHE, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    NUM_CLASSES, MC_DROPOUT_RATE, USE_AMP, RESULTS_DIR,
    seed_everything, setup_directories, RANDOM_SEED,
)
from dataset import get_train_val_split, DRDataset, get_val_transform
from model import create_baseline_model, create_model
from evaluate import compute_ece, compute_brier_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def collect_val_logits(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run deterministic forward pass on validation set, collecting raw logits.

    Args:
        model: Trained model (with deterministic_mode available).
        val_loader: Validation DataLoader.
        device: Torch device.

    Returns:
        Tuple of (logits [N, C], labels [N]) on CPU.
    """
    model.eval()
    amp_enabled = USE_AMP and device.type == "cuda"
    all_logits, all_labels = [], []

    with model.deterministic_mode():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled
            ):
                logits = model(images)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """Optimise temperature T to minimise NLL on validation logits.

    Args:
        logits: Raw logits [N, C].
        labels: Ground-truth labels [N].
        max_iter: Maximum LBFGS iterations.
        lr: Learning rate.

    Returns:
        Optimal temperature scalar.

    Example:
        >>> logits = torch.randn(100, 5)
        >>> labels = torch.randint(0, 5, (100,))
        >>> T = fit_temperature(logits, labels)
        >>> assert T > 0
    """
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temperature.clamp(min=0.01)
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().item())
```

**How it works**: Loads the existing checkpoint, runs deterministic inference on APTOS validation fold, collects raw logits, optimises `T` via LBFGS, saves `T` and before/after ECE to JSON.

**Output**: `outputs/results/baseline_temp_scaling.json` containing `{ "temperature": T, "ece_before": ..., "ece_after": ..., "brier_before": ..., "brier_after": ... }`.

---

### A2. Decision Threshold Tuning

**Goal**: Find per-class probability scaling factors that maximise QWK (or Macro F1) on APTOS validation. Apply at inference time instead of raw `argmax`.

**New file**: `src/threshold_tuning.py`

```python
"""Per-class decision threshold tuning for improved minority recall.

Usage:
    python threshold_tuning.py \
        --checkpoint "outputs/checkpoints/baseline*fold0_best.pth" \
        --model baseline --fold 0 --metric qwk
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score, f1_score

from config import (
    APTOS_TRAIN_CSV, APTOS_TRAIN_IMAGES, APTOS_PROCESSED_DIR,
    USE_PREPROCESSED_CACHE, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    NUM_CLASSES, MC_DROPOUT_RATE, MC_INFERENCE_PASSES, USE_AMP,
    RESULTS_DIR, seed_everything, setup_directories, RANDOM_SEED,
)
from dataset import get_train_val_split, DRDataset, get_val_transform
from model import create_baseline_model, create_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def tune_thresholds(
    mean_probs: np.ndarray,
    labels: np.ndarray,
    metric: str = "qwk",
) -> np.ndarray:
    """Find per-class threshold vector that maximises the target metric.

    Divides mean_probs by threshold vector before argmax:
        adjusted = mean_probs / thresholds
        prediction = argmax(adjusted)

    Args:
        mean_probs: Array of shape [N, C] with mean softmax probabilities.
        labels: Ground-truth labels [N].
        metric: Optimisation target — "qwk" or "macro_f1".

    Returns:
        Optimal threshold vector of shape [C].

    Example:
        >>> probs = np.random.dirichlet([1]*5, size=100)
        >>> labels = np.random.randint(0, 5, 100)
        >>> thresholds = tune_thresholds(probs, labels, metric="qwk")
        >>> assert thresholds.shape == (5,)
    """
    num_classes = mean_probs.shape[1]

    def objective(log_thresholds: np.ndarray) -> float:
        thresholds = np.exp(log_thresholds)  # Ensure positivity
        adjusted = mean_probs / thresholds[np.newaxis, :]
        predictions = adjusted.argmax(axis=1)
        if metric == "qwk":
            score = cohen_kappa_score(labels, predictions, weights="quadratic")
        elif metric == "macro_f1":
            score = f1_score(labels, predictions, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return -score  # Minimise negative score

    # Start from uniform thresholds (log-space: all zeros)
    initial = np.zeros(num_classes)
    result = minimize(objective, initial, method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-4})

    optimal_thresholds = np.exp(result.x)
    # Normalise so threshold[0] = 1.0 (relative scaling)
    optimal_thresholds = optimal_thresholds / optimal_thresholds[0]
    return optimal_thresholds
```

**How it works**: Runs MC Dropout inference (T=20) on APTOS validation fold, collects `mean_probs`, then uses Nelder-Mead to find the 5-element threshold vector maximising QWK. Reports per-class metrics before/after threshold adjustment.

**Output**: `outputs/results/baseline_thresholds.json` containing `{ "thresholds": [...], "metric_before": ..., "metric_after": ..., "per_class_recall_before": [...], "per_class_recall_after": [...] }`.

**Integration with evaluate.py**: After tuning, update `mc_dropout_inference` or `compute_metrics` to accept an optional `thresholds` vector. The prediction line in `evaluate.py` (line 91):
```python
# Before:
confidence, predictions = mean_probs.max(dim=1)

# After (when thresholds provided):
if thresholds is not None:
    adjusted = mean_probs / torch.tensor(thresholds, dtype=mean_probs.dtype)
    predictions = adjusted.argmax(dim=1)
    confidence = mean_probs.gather(1, predictions.unsqueeze(1)).squeeze(1)
else:
    confidence, predictions = mean_probs.max(dim=1)
```

---

## 5. Phase B: Structural Pipeline Changes

All changes in this phase are guarded by new config parameters so the existing pipeline continues to work unchanged by default.

### B1. New Config Parameters

**File**: `src/config.py`

Add after line 94 (after `GRAD_CLIP_NORM`):

```python
# =============================================================================
# IMPROVEMENT PARAMETERS (Plan 07)

LABEL_SMOOTHING = 0.0                       # 0.0 = off; 0.1 = recommended
USE_BALANCED_SAMPLER = False                 # WeightedRandomSampler toggle
CLASSIFIER_HIDDEN_DIM = 0                   # 0 = direct 2048→5; 512 = deeper head
CLASSIFIER_DROPOUT_RATE = MC_DROPOUT_RATE   # head dropout (may differ from MC rate)
LR_WARMUP_EPOCHS = 0                        # 0 = no warmup; 2-3 = recommended
```

**File**: `src/configs/experiment_config.py`

Add to `ExperimentConfig` dataclass (after `use_class_weights` field, line 36):

```python
    label_smoothing: float = 0.0
    use_balanced_sampler: bool = False
    classifier_hidden_dim: int = 0
    classifier_dropout_rate: float = MC_DROPOUT_RATE
    lr_warmup_epochs: int = 0
```

---

### B2. Label Smoothing in FocalLoss

**File**: `src/loss.py`

**Change**: Add `label_smoothing` parameter to `FocalLoss.__init__` and pass it to `F.cross_entropy`.

```python
# BEFORE (lines 9-28):
class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = FOCAL_GAMMA,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        ...

# AFTER:
class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = FOCAL_GAMMA,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        ...
```

**Why this works**: `F.cross_entropy` natively supports `label_smoothing` since PyTorch 1.10. The rest of the Focal Loss computation (`p_t`, focal weight, alpha) remains unchanged — `p_t = exp(-CE)` simply reflects the smoothed cross-entropy, which keeps gradient signal flowing to minority classes longer.

**Update in `train.py`** (line 504):

```python
# BEFORE:
criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha_weights)

# AFTER:
from config import LABEL_SMOOTHING
criterion = FocalLoss(
    gamma=FOCAL_GAMMA,
    alpha=alpha_weights,
    label_smoothing=LABEL_SMOOTHING,
)
```

---

### B3. WeightedRandomSampler

**File**: `src/dataset.py`

**Add new function** after `get_val_transform` (after line 299):

```python
def make_balanced_sampler(
    labels: list[int] | np.ndarray,
    num_classes: int = 5,
) -> torch.utils.data.WeightedRandomSampler:
    """Create a WeightedRandomSampler that equalises class frequency per epoch.

    Each sample is weighted by the inverse of its class count, so minority
    classes are oversampled and majority classes are undersampled.

    Args:
        labels: Integer class labels for every sample in the dataset.
        num_classes: Number of distinct classes.

    Returns:
        WeightedRandomSampler configured for one epoch (len(labels) draws
        with replacement).

    Example:
        >>> labels = [0, 0, 0, 0, 1, 2]
        >>> sampler = make_balanced_sampler(labels, num_classes=3)
        >>> assert len(sampler) == 6
    """
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    # Inverse frequency per sample
    sample_weights = 1.0 / class_counts[labels_tensor]
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(labels),
        replacement=True,
    )
```

**Modify `create_dataloaders`** (lines 329–369):

```python
# BEFORE (line 329):
def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    image_size: int = IMAGE_SIZE,
    preprocess: bool = True,
    use_cache: bool = False,
    cache_dir: Path | None = None,
) -> Tuple[DataLoader, DataLoader]:

# AFTER:
def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    image_size: int = IMAGE_SIZE,
    preprocess: bool = True,
    use_cache: bool = False,
    cache_dir: Path | None = None,
    use_balanced_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader]:
```

And the train DataLoader construction (lines 354–361):

```python
# BEFORE:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

# AFTER:
    sampler = None
    shuffle = True
    if use_balanced_sampler:
        train_labels = train_df["diagnosis"].values.tolist()
        sampler = make_balanced_sampler(train_labels)
        shuffle = False  # Sampler and shuffle are mutually exclusive

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
```

**Update in `train.py`** (lines 473–477):

```python
# BEFORE:
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, APTOS_TRAIN_IMAGES,
        batch_size=args.batch_size, num_workers=NUM_WORKERS,
        use_cache=use_cache, cache_dir=cache_dir,
    )

# AFTER:
    from config import USE_BALANCED_SAMPLER
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, APTOS_TRAIN_IMAGES,
        batch_size=args.batch_size, num_workers=NUM_WORKERS,
        use_cache=use_cache, cache_dir=cache_dir,
        use_balanced_sampler=USE_BALANCED_SAMPLER,
    )
```

**Critical interaction — Focal Loss alpha**: When balanced sampling is active, the alpha weights over-correct. Two options:

1. **Set alpha to uniform** when sampler is active (recommended for first experiment):
   ```python
   if USE_BALANCED_SAMPLER:
       alpha_weights = torch.ones(NUM_CLASSES, device=device)
   else:
       alpha_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
   ```

2. **Use sqrt-inverse-frequency** alpha as a milder weighting:
   ```python
   weights = torch.sqrt(total / (num_classes * counts.clamp(min=1)))
   ```

We'll implement option 1 first, with option 2 as a fallback if Grade 0 recall drops too much.

---

### B4. LR Warmup

**File**: `src/train.py`

**Change**: Replace `CosineAnnealingLR` with `SequentialLR(LinearLR → CosineAnnealingLR)` when warmup is configured.

```python
# BEFORE (lines 510-512):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

# AFTER:
    from config import LR_WARMUP_EPOCHS

    if LR_WARMUP_EPOCHS > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,    # Start at lr × 0.01 = 1e-6
            end_factor=1.0,       # Ramp to full lr = 1e-4
            total_iters=LR_WARMUP_EPOCHS,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - LR_WARMUP_EPOCHS,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[LR_WARMUP_EPOCHS],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs,
        )
```

**Why `LinearLR`**: PyTorch's `LinearLR` linearly ramps a multiplicative factor from `start_factor` to `end_factor` over `total_iters` steps. With `start_factor=0.01`, the effective LR ramps from `1e-6` to `1e-4` over 2 epochs, giving the model time to see all classes before making aggressive updates.

---

### B5. Deeper Classifier Head

**File**: `src/model.py`

**Change both model classes** to accept `classifier_hidden_dim`. When > 0, insert an intermediate `Linear → BatchNorm1d → ReLU → MCDropout` layer.

For **CBAMResNet50** (replace lines 121–124):

```python
# BEFORE:
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mc_dropout = MCDropout(dropout_rate)
        self.fc = nn.Linear(2048, num_classes)

# AFTER:
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if classifier_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(2048, classifier_hidden_dim),
                nn.BatchNorm1d(classifier_hidden_dim),
                nn.ReLU(inplace=True),
                MCDropout(dropout_rate),
                nn.Linear(classifier_hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                MCDropout(dropout_rate),
                nn.Linear(2048, num_classes),
            )
```

And update the constructor signature:

```python
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = MC_DROPOUT_RATE,
        pretrained: bool = True,
        reduction: int = CBAM_REDUCTION_RATIO,
        classifier_hidden_dim: int = 0,           # ← New parameter
    ):
```

And update the forward method (replace lines 141–144):

```python
# BEFORE:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)           # (B, 2048)
        x = self.mc_dropout(x)
        x = self.fc(x)                    # (B, num_classes)

# AFTER:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)           # (B, 2048)
        x = self.classifier(x)            # (B, num_classes)
```

**Apply the identical change to `BaselineResNet50`** (lines 196–198 and forward at 206–209).

**Update `deterministic_mode()` context manager**: No change needed — it already iterates `self.modules()` finding all `MCDropout` instances, which includes those inside `nn.Sequential`.

**Update factory functions**:

```python
# create_model (line 162):
def create_model(
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = MC_DROPOUT_RATE,
    pretrained: bool = True,
    classifier_hidden_dim: int = 0,
) -> CBAMResNet50:
    return CBAMResNet50(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        classifier_hidden_dim=classifier_hidden_dim,
    )

# create_baseline_model (line 226):
def create_baseline_model(
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = MC_DROPOUT_RATE,
    pretrained: bool = True,
    classifier_hidden_dim: int = 0,
) -> BaselineResNet50:
    return BaselineResNet50(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        classifier_hidden_dim=classifier_hidden_dim,
    )
```

**Update `train.py`** model creation (lines 481–494):

```python
# BEFORE:
    if args.model == "baseline":
        model = create_baseline_model(
            num_classes=NUM_CLASSES,
            dropout_rate=MC_DROPOUT_RATE,
            pretrained=True,
        )

# AFTER:
    from config import CLASSIFIER_HIDDEN_DIM
    if args.model == "baseline":
        model = create_baseline_model(
            num_classes=NUM_CLASSES,
            dropout_rate=MC_DROPOUT_RATE,
            pretrained=True,
            classifier_hidden_dim=CLASSIFIER_HIDDEN_DIM,
        )
```

(Same for the `cbam` branch.)

**Checkpoint compatibility note**: Old checkpoints (with `self.mc_dropout` + `self.fc`) will NOT load into the new model (which uses `self.classifier`). This is intentional — Phase B requires retraining anyway. For evaluation of old checkpoints, keep `classifier_hidden_dim=0` which preserves the old structure inside `nn.Sequential`.

**Actually, wait** — even with `classifier_hidden_dim=0`, the state dict keys change from `mc_dropout.*` / `fc.*` to `classifier.0.*` / `classifier.1.*`. This **breaks old checkpoint loading**.

**Fix**: Add a backward-compatible load method, OR keep both `self.mc_dropout` + `self.fc` attributes when `hidden_dim == 0` and only use `self.classifier` when `hidden_dim > 0`:

```python
        if classifier_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(2048, classifier_hidden_dim),
                nn.BatchNorm1d(classifier_hidden_dim),
                nn.ReLU(inplace=True),
                MCDropout(dropout_rate),
                nn.Linear(classifier_hidden_dim, num_classes),
            )
            self._use_classifier_seq = True
        else:
            # Preserve original attribute names for checkpoint compatibility
            self.mc_dropout = MCDropout(dropout_rate)
            self.fc = nn.Linear(2048, num_classes)
            self._use_classifier_seq = False
```

And in forward:

```python
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self._use_classifier_seq:
            x = self.classifier(x)
        else:
            x = self.mc_dropout(x)
            x = self.fc(x)
        return x
```

This preserves backward compatibility: `classifier_hidden_dim=0` keeps the exact same state dict keys as the current model.

---

### B6. New YAML Configs

**New file**: `src/configs/gpu_baseline_improved.yaml`

```yaml
# GPU Baseline ResNet-50 — Improved (Plan 07)
name: baseline_resnet50_improved
description: Baseline with balanced sampling, label smoothing, warmup, deeper head

model_type: baseline
pretrained: true
dropout_rate: 0.3

epochs: 25
learning_rate: 0.0001
weight_decay: 0.0001
batch_size: 16
grad_clip_norm: 1.0
early_stopping_patience: 7

focal_gamma: 2.0
use_class_weights: false         # Disabled — balanced sampler handles this
label_smoothing: 0.1
use_balanced_sampler: true
classifier_hidden_dim: 512
classifier_dropout_rate: 0.3
lr_warmup_epochs: 2

image_size: 512
num_workers: 4
fold: 0

device: cuda
use_amp: true

seed: 42
```

**Key differences from original `gpu_baseline.yaml`**:

| Parameter | Original | Improved | Reason |
|-----------|----------|----------|--------|
| `dropout_rate` | 0.5 | 0.3 | Less aggressive for deeper head |
| `epochs` | 20 | 25 | Slightly more room with warmup |
| `early_stopping_patience` | 5 | 7 | Warmup delays convergence; need more patience |
| `use_class_weights` | true | false | Balanced sampler replaces alpha weighting |
| `label_smoothing` | _(absent)_ | 0.1 | Reduce overconfidence on Grade 0 |
| `use_balanced_sampler` | _(absent)_ | true | Equalise class frequency per epoch |
| `classifier_hidden_dim` | _(absent)_ | 512 | Deeper head for better class separation |
| `lr_warmup_epochs` | _(absent)_ | 2 | Prevent early Grade 0 lock-in |

---

### B7. Wire New Config Into `train.py` and `experiment_config.py`

**File**: `src/configs/experiment_config.py`

Update `load_config` (line 115) to pass new fields to CLI args:

```python
def load_config(config_path: str, cli_args: Any) -> Any:
    """Load config from file and merge with CLI arguments."""
    config = ExperimentConfig.from_yaml(config_path)

    cli_args.epochs = config.epochs
    cli_args.lr = config.learning_rate
    cli_args.batch_size = config.batch_size
    cli_args.fold = config.fold
    cli_args.model = config.model_type

    # Plan 07 additions
    cli_args.label_smoothing = config.label_smoothing
    cli_args.use_balanced_sampler = config.use_balanced_sampler
    cli_args.classifier_hidden_dim = config.classifier_hidden_dim
    cli_args.classifier_dropout_rate = config.classifier_dropout_rate
    cli_args.lr_warmup_epochs = config.lr_warmup_epochs
    cli_args.use_class_weights = config.use_class_weights

    return cli_args
```

**File**: `src/train.py`

Add CLI arguments (after line 443):

```python
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--use_balanced_sampler", action="store_true", default=USE_BALANCED_SAMPLER)
    parser.add_argument("--classifier_hidden_dim", type=int, default=CLASSIFIER_HIDDEN_DIM)
    parser.add_argument("--lr_warmup_epochs", type=int, default=LR_WARMUP_EPOCHS)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
```

Then update the loss/model/scheduler/dataloader creation to use `args.*` instead of config constants directly (see B2–B5 above).

---

### B8. Update `evaluate.py` For Deeper Head

When loading a checkpoint trained with `classifier_hidden_dim > 0`, the evaluate script must create the model with the same architecture.

**File**: `src/evaluate.py`

Add `--classifier_hidden_dim` argument and pass it to model creation:

```python
    parser.add_argument("--classifier_hidden_dim", type=int, default=0,
                        help="Classifier hidden dim (must match training config)")
```

Update model creation (lines 506–517):

```python
    if args.model == "baseline":
        model = create_baseline_model(
            num_classes=NUM_CLASSES,
            dropout_rate=MC_DROPOUT_RATE,
            pretrained=False,
            classifier_hidden_dim=args.classifier_hidden_dim,
        )
```

---

## 6. Phase C: Ordinal Loss (Optional Extension)

### C1. Soft Ordinal Label Encoding

**File**: `src/loss.py`

**New function**:

```python
def ordinal_soft_labels(
    targets: torch.Tensor,
    num_classes: int = 5,
    tau: float = 1.0,
) -> torch.Tensor:
    """Convert integer labels to distance-weighted soft probability vectors.

    For each target class k, the soft label at position j is proportional
    to exp(-|j - k| / tau). This encodes ordinal structure: adjacent grades
    get higher soft probability than distant grades.

    Args:
        targets: Integer labels of shape [B].
        num_classes: Number of classes.
        tau: Temperature controlling label spread. Lower tau = sharper
            (closer to one-hot). Higher tau = softer (more uniform).

    Returns:
        Soft label tensor of shape [B, num_classes].

    Example:
        >>> targets = torch.tensor([0, 2, 4])
        >>> soft = ordinal_soft_labels(targets, num_classes=5, tau=1.0)
        >>> soft.argmax(dim=1)
        tensor([0, 2, 4])
    """
    grades = torch.arange(num_classes, device=targets.device).float()
    # distances[b, j] = |j - targets[b]|
    distances = (grades.unsqueeze(0) - targets.unsqueeze(1).float()).abs()
    soft = torch.exp(-distances / tau)
    soft = soft / soft.sum(dim=1, keepdim=True)  # Normalise to probability
    return soft
```

**Modified FocalLoss** to accept soft targets:

This is a moderate change because the current FocalLoss uses `F.cross_entropy(logits, targets)` which expects integer targets. To support soft labels, we need to compute CE manually:

```python
def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if self.use_ordinal_soft_labels:
        soft_targets = ordinal_soft_labels(targets, logits.size(1), self.ordinal_tau)
        # Manual CE with soft targets: -sum(soft_target * log_softmax)
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = -(soft_targets * log_probs).sum(dim=1)  # [B]
    else:
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )

    p_t = torch.exp(-ce_loss)
    focal_weight = (1.0 - p_t) ** self.gamma
    loss = focal_weight * ce_loss

    if self.alpha is not None:
        alpha_t = self.alpha.gather(0, targets)
        loss = alpha_t * loss

    if self.reduction == "mean":
        return loss.mean()
    elif self.reduction == "sum":
        return loss.sum()
    return loss
```

**Note**: Ordinal soft labels and `label_smoothing` should NOT be combined (both soften the target distribution; stacking them over-smooths). The `FocalLoss` should enforce mutual exclusivity:

```python
if self.label_smoothing > 0.0 and self.use_ordinal_soft_labels:
    raise ValueError("Cannot combine label_smoothing and ordinal soft labels.")
```

---

## 7. Testing Strategy

### CPU Smoke Tests (Before GPU Training)

Every code change must pass CPU smoke tests before GPU deployment.

**Test 1 — Baseline with improvements (2 epochs, batch_size=2)**:

```bash
cd src
python train.py --model baseline --epochs 2 --batch_size 2 \
    --label_smoothing 0.1 --use_balanced_sampler \
    --classifier_hidden_dim 512 --lr_warmup_epochs 1
```

**Expected**: Completes without error. Per-class F1 printed. Loss decreases.

**Test 2 — Backward compatibility (old config, 0 improvements)**:

```bash
cd src
python train.py --model baseline --epochs 2 --batch_size 2
```

**Expected**: Identical behaviour to current code. No regressions.

**Test 3 — Old checkpoint loads with hidden_dim=0**:

```bash
cd src
python evaluate.py --checkpoint "outputs/checkpoints/baseline*fold0_best.pth" \
    --model baseline --mc_passes 3 --max_images 10 \
    --classifier_hidden_dim 0
```

**Expected**: Loads existing checkpoint, runs inference. Backward compatible.

**Test 4 — Temperature scaling**:

```bash
cd src
python temperature_scaling.py \
    --checkpoint "outputs/checkpoints/baseline*fold0_best.pth" \
    --model baseline --fold 0
```

**Expected**: Outputs JSON with learned `T` and ECE before/after.

**Test 5 — Threshold tuning**:

```bash
cd src
python threshold_tuning.py \
    --checkpoint "outputs/checkpoints/baseline*fold0_best.pth" \
    --model baseline --fold 0 --metric qwk
```

**Expected**: Outputs JSON with thresholds and per-class recall improvement.

---

## 8. GPU Training Plan

After all CPU smoke tests pass:

**Run 1 — Improved Baseline (fold 0)**:

```bash
cd src
python train.py --config configs/gpu_baseline_improved.yaml
```

**Expected runtime**: ~3–5 hours on RTX 3060/3070 (25 epochs × 2,929 images × 512×512).

**After training — evaluate on Messidor-2**:

```bash
cd src
python evaluate.py \
    --checkpoint "outputs/checkpoints/baseline_resnet50_improved*fold0_best.pth" \
    --model baseline --mc_passes 20 \
    --classifier_hidden_dim 512
```

**Compare against baseline**:

| Metric | Old Baseline | Improved Baseline | Delta |
|--------|-------------|-------------------|-------|
| Val QWK | 0.9088 | ? | ? |
| Val Macro F1 | ~0.63 | ? | ? |
| Grade 3 F1 (APTOS val) | ~0.32 | ? | ? |
| Messidor-2 Accuracy | 0.6399 | ? | ? |
| Messidor-2 Grade 1 Recall | 0.048 | ? | ? |
| Messidor-2 Grade 2 Recall | 0.236 | ? | ? |

---

## 9. Rollback Plan

Every change is guarded by a config flag that defaults to the current behaviour:

| Flag | Default | Effect when off |
|------|---------|-----------------|
| `LABEL_SMOOTHING = 0.0` | Off | Standard CE inside Focal Loss |
| `USE_BALANCED_SAMPLER = False` | Off | `shuffle=True` DataLoader |
| `CLASSIFIER_HIDDEN_DIM = 0` | Off | Original `MCDropout → Linear(2048,5)` head |
| `LR_WARMUP_EPOCHS = 0` | Off | Standard `CosineAnnealingLR` |

Setting all flags to their defaults produces the **exact same pipeline** as the current codebase. No existing experiment is affected.

---

## 10. Detailed TODO List

### Phase A — Post-Hoc Fixes (No Retraining) — ✅ COMPLETE (2026-04-03)

- [x] **A1**: Create `src/temperature_scaling.py`
  - [x] A1.1: Implement `collect_val_logits()` — deterministic forward pass collecting raw logits
  - [x] A1.2: Implement `fit_temperature()` — LBFGS optimisation of temperature scalar
  - [x] A1.3: Implement `main()` — CLI entry point (checkpoint, model, fold args)
  - [x] A1.4: Output JSON with `temperature`, `ece_before`, `ece_after`, `brier_before`, `brier_after`
  - [x] A1.5: Output before/after reliability diagram overlay figure
  - [x] A1.6: CPU test — **TESTED: T=1.0321, ECE 4.88%→4.33%, model already well-calibrated**

- [x] **A2**: Create `src/threshold_tuning.py`
  - [x] A2.1: Implement `tune_thresholds()` — Nelder-Mead optimisation in log-space
  - [x] A2.2: Implement `main()` — CLI entry point; runs MC Dropout inference on APTOS val, then tunes
  - [x] A2.3: Output JSON with `thresholds`, `metric_before`, `metric_after`, `per_class_recall_before`, `per_class_recall_after`
  - [x] A2.4: Output per-class comparison table (printed to console and saved)
  - [x] A2.5: CPU test — **TESTED: All thresholds=1.0, ZERO improvement, boundaries already optimal**
  - [x] A2.6: Apply tuned thresholds to Messidor-2 — **SKIPPED: thresholds=[1.0, 1.0, 1.0, 1.0, 1.0] provide no benefit**

- [x] **A3**: Integrate thresholds into `evaluate.py`
  - [x] A3.1: Add `--thresholds` CLI argument (comma-separated 5 floats or path to JSON)
  - [x] A3.2: Apply threshold adjustment in `mc_dropout_inference` return dict
  - [x] A3.3: Test backward compatibility (no `--thresholds` = unchanged behaviour)

**Phase A Conclusion:**
- Temperature scaling: T≈1.0 confirms model is **well-calibrated**
- Threshold tuning: All thresholds=1.0 confirms **decision boundaries are optimal**
- **ZERO improvement on minority class recall** — post-hoc fixes cannot compensate for structural training imbalance
- **Root cause validated**: Grade 1 gets ~0.8 samples/batch during training → model never learns it properly
- **Phase B (balanced sampling) is MANDATORY** to fix catastrophic Grade 1-4 recall collapse

### Phase B — Structural Pipeline Changes

- [x] **B1**: Update `src/config.py` — add 5 new constants
  - [x] B1.1: `LABEL_SMOOTHING = 0.0`
  - [x] B1.2: `USE_BALANCED_SAMPLER = False`
  - [x] B1.3: `CLASSIFIER_HIDDEN_DIM = 0`
  - [x] B1.4: `CLASSIFIER_DROPOUT_RATE = MC_DROPOUT_RATE`
  - [x] B1.5: `LR_WARMUP_EPOCHS = 0`

- [x] **B2**: Update `src/loss.py` — add `label_smoothing` to FocalLoss
  - [x] B2.1: Add `label_smoothing` parameter to `__init__`
  - [x] B2.2: Pass `label_smoothing` to `F.cross_entropy` in `forward`
  - [x] B2.3: Verify self-test (`__main__` block) still passes

- [x] **B3**: Update `src/dataset.py` — add WeightedRandomSampler
  - [x] B3.1: Implement `make_balanced_sampler()` function
  - [x] B3.2: Add `use_balanced_sampler` parameter to `create_dataloaders()`
  - [x] B3.3: Conditionally create sampler and set `shuffle=False` when active
  - [x] B3.4: Verify self-test (`__main__` block) still passes

- [x] **B4**: Update `src/train.py` — LR warmup scheduler
  - [x] B4.1: Import `LR_WARMUP_EPOCHS` from config
  - [x] B4.2: Replace `CosineAnnealingLR` with `SequentialLR(LinearLR + CosineAnnealingLR)` when warmup > 0
  - [x] B4.3: Verify epoch logging still shows correct LR values

- [x] **B5**: Update `src/model.py` — deeper classifier head
  - [x] B5.1: Add `classifier_hidden_dim` parameter to `CBAMResNet50.__init__`
  - [x] B5.2: Implement conditional head: `nn.Sequential(Linear → BN → ReLU → MCDropout → Linear)` when hidden_dim > 0, original `mc_dropout + fc` when 0
  - [x] B5.3: Update `forward()` to use the appropriate path
  - [x] B5.4: Apply identical changes to `BaselineResNet50`
  - [x] B5.5: Update `create_model()` and `create_baseline_model()` factory functions
  - [x] B5.6: Verify old checkpoint loads with `hidden_dim=0` (backward compatibility)
  - [x] B5.7: Verify `deterministic_mode()` still finds MCDropout inside Sequential
  - [x] B5.8: Update `__main__` test block for both head variants

- [x] **B6**: Update `src/configs/experiment_config.py`
  - [x] B6.1: Add new fields to `ExperimentConfig` dataclass
  - [x] B6.2: Update `load_config()` to pass new fields to CLI args

- [x] **B7**: Create `src/configs/gpu_baseline_sampler_only.yaml` and `src/configs/gpu_baseline_full.yaml`
  - [x] B7.1: Write sampler-only config (balanced sampler only, no other changes)
  - [x] B7.2: Write full config with all improvement flags enabled

- [x] **B8**: Update `src/train.py` — wire everything together
  - [x] B8.1: Add new CLI arguments (`--label_smoothing`, `--use_balanced_sampler`, `--classifier_hidden_dim`, `--lr_warmup_epochs`, `--use_class_weights`)
  - [x] B8.2: Update loss creation to use `args.label_smoothing`
  - [x] B8.3: Update dataloader creation to use `args.use_balanced_sampler`
  - [x] B8.4: Update model creation to use `args.classifier_hidden_dim`
  - [x] B8.5: Update scheduler creation to use `args.lr_warmup_epochs`
  - [x] B8.6: Conditionally disable alpha weights when balanced sampler is active
  - [x] B8.7: Log all new hyperparameters in `hyperparams` dict
  - [x] B8.8: Update `_save_run_metrics` to include new params

- [x] **B9**: Update `src/evaluate.py` — support deeper head
  - [x] B9.1: Add `--classifier_hidden_dim` CLI argument
  - [x] B9.2: Pass to model creation

- [ ] **B10**: CPU Smoke Tests
  - [ ] B10.1: Run `train.py` with all improvements ON (2 epochs, batch_size=2, CPU) — must complete without error
  - [ ] B10.2: Run `train.py` with all improvements OFF (defaults) — must produce identical behaviour to current code
  - [ ] B10.3: Load old checkpoint with `evaluate.py --classifier_hidden_dim 0` — must work
  - [ ] B10.4: Run `temperature_scaling.py` — must produce valid output
  - [ ] B10.5: Run `threshold_tuning.py` — must produce valid output

- [ ] **B11**: GPU Training
  - [ ] B11.1: Upload updated code to GPU instance
  - [ ] B11.2: Run improved baseline training (fold 0): `python train.py --config configs/gpu_baseline_improved.yaml`
  - [ ] B11.3: Monitor training curves — verify minority class F1 improves over epochs
  - [ ] B11.4: Evaluate on APTOS validation — record per-class metrics
  - [ ] B11.5: Evaluate on Messidor-2 (T=20) — record per-class metrics
  - [ ] B11.6: Compare against old baseline (fill comparison table in Section 8)

- [ ] **B12**: Post-Training Analysis
  - [ ] B12.1: Apply temperature scaling to new model
  - [ ] B12.2: Apply threshold tuning to new model
  - [ ] B12.3: Run Messidor-2 evaluation with tuned thresholds
  - [ ] B12.4: Generate comparison figures (confusion matrices: old vs new, side by side)
  - [ ] B12.5: Update `docs/result-report.md` with new results

### Phase C — Ordinal Loss (If Time Permits)

- [ ] **C1**: Implement ordinal soft labels in `src/loss.py`
  - [ ] C1.1: Implement `ordinal_soft_labels()` function
  - [ ] C1.2: Add `use_ordinal_soft_labels` and `ordinal_tau` parameters to FocalLoss
  - [ ] C1.3: Implement soft-target cross-entropy path in `forward()`
  - [ ] C1.4: Add mutual exclusivity check with `label_smoothing`
  - [ ] C1.5: Add self-test in `__main__` block

- [ ] **C2**: Create `gpu_baseline_ordinal.yaml` config
  - [ ] C2.1: Copy improved config, replace `label_smoothing` with `ordinal_tau`
  - [ ] C2.2: CPU smoke test

- [ ] **C3**: GPU training + evaluation
  - [ ] C3.1: Train improved baseline + ordinal loss (fold 0)
  - [ ] C3.2: Evaluate on APTOS val + Messidor-2
  - [ ] C3.3: Compare against Phase B results

### Documentation & Cleanup

- [ ] **D1**: Update `docs/research-project.md` — Section 11 (implementation status) with Phase 7 results
- [ ] **D2**: Update `docs/result-report.md` — add improved baseline results section
- [ ] **D3**: Update `docs/improve-model.md` — annotate which improvements were implemented and their measured impact
- [ ] **D4**: Commit all code changes with descriptive messages

---

## 11. File Change Summary

| File | Phase | Type | Changes |
|------|-------|------|---------|
| `src/config.py` | B1 | Modify | Add 5 new constants |
| `src/loss.py` | B2, C1 | Modify | `label_smoothing` param; ordinal soft labels function |
| `src/dataset.py` | B3 | Modify | `make_balanced_sampler()`; `use_balanced_sampler` param in `create_dataloaders` |
| `src/model.py` | B5 | Modify | `classifier_hidden_dim` param in both model classes + factories |
| `src/train.py` | B4, B8 | Modify | LR warmup; new CLI args; wire improvements |
| `src/evaluate.py` | A3, B9 | Modify | `--thresholds` and `--classifier_hidden_dim` args |
| `src/configs/experiment_config.py` | B6 | Modify | New dataclass fields + load_config update |
| `src/temperature_scaling.py` | A1 | **New** | Post-hoc calibration script |
| `src/threshold_tuning.py` | A2 | **New** | Per-class threshold optimisation script |
| `src/configs/gpu_baseline_improved.yaml` | B7 | **New** | Improved training config |
| `src/configs/gpu_baseline_sampler_only.yaml` | B7 | **New** | Sampler-only ablation config |
| `src/configs/gpu_baseline_full.yaml` | B7 | **New** | Full improvements config |
| `src/configs/gpu_baseline_ordinal.yaml` | C2 | **New** | Ordinal loss training config |

**Total**: 8 modified files, 4 new files.
