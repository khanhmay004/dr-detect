# Implementation Plan: Phase 0 Infrastructure Fixes + Phase 1 Baseline Retraining

> **Document Version**: 1.0
> **Created**: 2026-03-30
> **Scope**: Phase 0 (Infrastructure Fixes) + Phase 1 (Baseline Retraining)
> **Target Directory**: `src/` (extend existing modules)

---

## Executive Summary 

This plan addresses critical code-level issues that affect experiment validity (Phase 0) and establishes a clean, properly-controlled baseline for ablation study (Phase 1). The goal is to ensure all subsequent experiments (CBAM model, ablation variants) share an identical training pipeline—the only variable being the architecture itself.

**Key deliverables**:
1. Fixed MCDropout with deterministic validation mode
2. Gradient clipping for Focal Loss training stability
3. Unified config system supporting CPU test runs and GPU full training
4. Baseline ResNet-50 model trained with identical pipeline to CBAM model
5. Messidor-2 external validation of baseline

---

## Table of Contents

1. [Phase 0: Infrastructure Fixes](#phase-0-infrastructure-fixes)
   - [0.1 Fix MCDropout Deterministic Validation](#01-fix-mcdropout-deterministic-validation)
   - [0.2 Add Gradient Clipping](#02-add-gradient-clipping)
   - [0.3 Fix Progress Bar Epoch Display](#03-fix-progress-bar-epoch-display)
   - [0.4 Add Baseline Model Factory](#04-add-baseline-model-factory)
   - [0.5 Add --model CLI Argument](#05-add---model-cli-argument)
   - [0.6 Create Experiment Config System](#06-create-experiment-config-system)
2. [Phase 1: Baseline Retraining](#phase-1-baseline-retraining)
   - [1.1 CPU Smoke Test](#11-cpu-smoke-test)
   - [1.2 GPU Full Training](#12-gpu-full-training)
   - [1.3 Messidor-2 Evaluation](#13-messidor-2-evaluation)
3. [Directory Structure](#directory-structure)
4. [Detailed TODO Checklist](#detailed-todo-checklist)
5. [Verification Criteria](#verification-criteria)
6. [Risk Mitigation](#risk-mitigation)

---

## Phase 0: Infrastructure Fixes

### 0.1 Fix MCDropout Deterministic Validation

**Problem**: The current `MCDropout` class hard-codes `training=True` in `F.dropout()`, meaning validation during training is stochastic. This introduces noise into:
- Early stopping decisions (kappa-based)
- Best model selection (kappa tracking)
- Validation loss curves (non-reproducible)

**Impact**: The "best" model checkpoint might be the result of a lucky dropout mask, not genuinely superior weights.

**Solution**: Add an `mc_active` toggle that can be temporarily disabled during training validation, while remaining active during MC Dropout inference.

**File**: `src/model.py`

```python
# ==========================================================================
# UPDATED MCDropout class with deterministic mode support
# ==========================================================================

from contextlib import contextmanager

class MCDropout(nn.Module):
    """Dropout layer that stays active during model.eval().

    Can be temporarily disabled via the mc_active flag for deterministic
    validation during training (as opposed to MC inference at test time).

    Attributes:
        p: Dropout probability.
        mc_active: When True (default), dropout is always applied.
                   When False, acts as identity (for deterministic validation).
    """

    def __init__(self, p: float = MC_DROPOUT_RATE):
        super().__init__()
        self.p = p
        self.mc_active = True  # Toggle for training validation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mc_active:
            return F.dropout(x, self.p, training=True)
        return x


class CBAMResNet50(nn.Module):
    # ... existing __init__ and forward ...

    @contextmanager
    def deterministic_mode(self):
        """Temporarily disable MC Dropout for deterministic validation.

        Usage:
            with model.deterministic_mode():
                logits = model(images)  # no dropout stochasticity

        This context manager finds all MCDropout modules in the model and
        disables their stochastic behavior. On exit, it re-enables them.
        """
        mc_modules = [m for m in self.modules() if isinstance(m, MCDropout)]
        for m in mc_modules:
            m.mc_active = False
        try:
            yield
        finally:
            for m in mc_modules:
                m.mc_active = True
```

**File**: `src/train.py` — Update `Trainer.validate()`

```python
@torch.no_grad()
def validate(self, val_loader, criterion):
    """Validate with deterministic (non-MC) inference."""
    self.model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Val]",
    )
    for images, labels in pbar:
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # Use deterministic mode for validation during training
        with self.model.deterministic_mode():
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.amp_enabled
            ):
                logits = self.model(images)
                loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # ... rest of validation metrics computation ...
```

**Verification Test**:
```python
# Test script: verify deterministic validation
model = CBAMResNet50(pretrained=False)
model.eval()
x = torch.randn(2, 3, 512, 512)

# MC mode (default): outputs differ
o1, o2 = model(x), model(x)
assert not torch.equal(o1, o2), "MC should produce different outputs"

# Deterministic mode: outputs identical
with model.deterministic_mode():
    d1 = model(x)
    d2 = model(x)
assert torch.equal(d1, d2), "Deterministic mode should produce identical outputs"
print("MCDropout deterministic mode test passed!")
```

---

### 0.2 Add Gradient Clipping

**Problem**: Focal Loss with γ=2.0 amplifies gradients on hard/misclassified examples. In APTOS 2019, some "hard" examples are actually mislabeled (single Kaggle grader), leading to potentially explosive gradients on noisy samples.

**Solution**: Add gradient clipping before the optimizer step.

**File**: `src/train.py` — Update `Trainer.train_epoch()`

```python
def train_epoch(self, train_loader, criterion, optimizer):
    """Train for one epoch with AMP and gradient clipping."""
    self.model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]",
    )
    for images, labels in pbar:
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # --- AMP forward ---
        with torch.amp.autocast(
            device_type=self.device.type, enabled=self.amp_enabled
        ):
            logits = self.model(images)
            loss = criterion(logits, labels)

        # --- AMP backward with gradient clipping ---
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)  # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.grad_clip_norm
        )
        self.scaler.step(optimizer)
        self.scaler.update()

        # ... rest of bookkeeping ...
```

**Config Addition** (`src/config.py`):
```python
# Gradient clipping
GRAD_CLIP_NORM = 1.0  # Max L2 norm for gradient clipping
```

**Trainer `__init__` update**:
```python
def __init__(self, model, device, fold=0, grad_clip_norm=GRAD_CLIP_NORM):
    # ... existing code ...
    self.grad_clip_norm = grad_clip_norm
```

---

### 0.3 Fix Progress Bar Epoch Display

**Problem**: Progress bars show `{current}/{EPOCHS}` using the config constant, not the actual `num_epochs` argument. If `--epochs 30` is passed, the display still shows `/20`.

**Solution**: Store `num_epochs` in Trainer and use it in tqdm descriptions.

**File**: `src/train.py`

```python
class Trainer:
    def __init__(self, model, device, fold=0, grad_clip_norm=GRAD_CLIP_NORM):
        # ... existing initialization ...
        self.num_epochs = EPOCHS  # Default, updated in fit()

    def fit(self, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=EPOCHS):
        """Full training loop with early stopping."""
        self.num_epochs = num_epochs  # Store actual value
        start_epoch = self.current_epoch
        # ... rest of fit ...

    def train_epoch(self, train_loader, criterion, optimizer):
        # Use self.num_epochs instead of EPOCHS constant
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]",
        )
        # ...

    def validate(self, val_loader, criterion):
        # Use self.num_epochs instead of EPOCHS constant
        pbar = tqdm(
            val_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Val]",
        )
        # ...
```

---

### 0.4 Add Baseline Model Factory

**Problem**: The current baseline (from notebook) was trained with different hyperparameters (batch_size=4, CPU, 5 epochs). This confounds the ablation comparison—we cannot isolate CBAM's contribution.

**Solution**: Create a `create_baseline_model()` factory that uses the **exact same head** (MCDropout + Linear) as `CBAMResNet50`, ensuring the only difference is the attention mechanism.

**File**: `src/model.py`

```python
# ==========================================================================
# Baseline ResNet-50 without CBAM (for ablation study)
# ==========================================================================

class BaselineResNet50(nn.Module):
    """Baseline ResNet-50 without CBAM attention.

    Uses the same classification head (MCDropout + Linear) as CBAMResNet50
    to ensure a fair ablation comparison. The ONLY difference between this
    and CBAMResNet50 is the presence/absence of CBAM modules.

    Architecture::

        Input (B, 3, 512, 512)
          │
          ├─ ResNet-50 backbone (stem + layer1-4)
          │
          ├─ AdaptiveAvgPool2d → (B, 2048)
          ├─ MCDropout(p=0.5)          ← same as CBAM model
          └─ Linear(2048, 5)

    Args:
        num_classes: Output logits dimension (5 for DR grading).
        dropout_rate: MC Dropout probability.
        pretrained: Use ImageNet-pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = MC_DROPOUT_RATE,
        pretrained: bool = True,
    ):
        super().__init__()

        # --- Backbone ---
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Extract stages (exclude original fc and avgpool)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # --- Classification head (identical to CBAM model) ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mc_dropout = MCDropout(dropout_rate)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (B, num_classes)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mc_dropout(x)
        x = self.fc(x)
        return x

    @contextmanager
    def deterministic_mode(self):
        """Temporarily disable MC Dropout for deterministic validation."""
        mc_modules = [m for m in self.modules() if isinstance(m, MCDropout)]
        for m in mc_modules:
            m.mc_active = False
        try:
            yield
        finally:
            for m in mc_modules:
                m.mc_active = True


def create_baseline_model(
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = MC_DROPOUT_RATE,
    pretrained: bool = True,
) -> BaselineResNet50:
    """Factory function for baseline model (no CBAM)."""
    return BaselineResNet50(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
    )
```

---

### 0.5 Add --model CLI Argument

**Problem**: `train.py` currently only supports CBAM model. We need to train both baseline and CBAM with identical pipelines.

**Solution**: Add `--model {baseline,cbam}` CLI argument.

**File**: `src/train.py`

```python
from model import create_model, create_baseline_model

def main():
    parser = argparse.ArgumentParser(description="Train DR detection model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fold", type=int, default=0, help="Validation fold (0-4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    # NEW: Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="cbam",
        choices=["baseline", "cbam"],
        help="Model architecture: 'baseline' (ResNet-50) or 'cbam' (CBAM-ResNet50)"
    )
    # NEW: Config file support
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides CLI args)"
    )
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        args = load_config(args.config, args)

    # ... existing setup code ...

    # ---- Model Selection ----
    print(f"\nBuilding {args.model.upper()} model ...")
    if args.model == "baseline":
        model = create_baseline_model(
            num_classes=NUM_CLASSES,
            dropout_rate=MC_DROPOUT_RATE,
            pretrained=True,
        )
        model_name = "baseline_resnet50"
    else:  # cbam
        model = create_model(
            num_classes=NUM_CLASSES,
            dropout_rate=MC_DROPOUT_RATE,
            pretrained=True,
        )
        model_name = "cbam_resnet50"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ... rest of training setup ...

    # Update checkpoint names based on model type
    trainer = Trainer(model, device, fold=args.fold, model_name=model_name)
```

**Trainer update for dynamic checkpoint naming**:

```python
class Trainer:
    def __init__(self, model, device, fold=0, model_name="cbam_resnet50",
                 grad_clip_norm=GRAD_CLIP_NORM):
        # ... existing code ...
        self.model_name = model_name

    def save_checkpoint(self, optimizer, scheduler, is_best=False):
        checkpoint = { ... }

        last_path = CHECKPOINT_DIR / f"{self.model_name}_fold{self.fold}_last.pth"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = CHECKPOINT_DIR / f"{self.model_name}_fold{self.fold}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"  New best model saved (kappa = {self.best_kappa:.4f})")

    def _save_history(self):
        path = LOG_DIR / f"{self.model_name}_fold{self.fold}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  History saved to {path}")
```

---

### 0.6 Create Experiment Config System

**Purpose**: Enable easy switching between CPU smoke tests and GPU full training without CLI argument juggling.

**File**: `src/configs/experiment_config.py`

```python
"""
Experiment configuration system.

Supports:
- YAML config files for experiment presets
- CPU smoke test mode for code verification
- GPU full training mode for actual experiments
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_WORKERS,
    IMAGE_SIZE, MC_DROPOUT_RATE, FOCAL_GAMMA, GRAD_CLIP_NORM,
    RANDOM_SEED, N_FOLDS, EARLY_STOPPING_PATIENCE,
)


@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment."""

    # Experiment metadata
    name: str = "default"
    description: str = ""

    # Model
    model_type: str = "cbam"  # "baseline" or "cbam"
    pretrained: bool = True
    dropout_rate: float = MC_DROPOUT_RATE

    # Training
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    batch_size: int = BATCH_SIZE
    grad_clip_norm: float = GRAD_CLIP_NORM
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE

    # Loss
    focal_gamma: float = FOCAL_GAMMA
    use_class_weights: bool = True

    # Data
    image_size: int = IMAGE_SIZE
    num_workers: int = NUM_WORKERS
    fold: int = 0
    n_folds: int = N_FOLDS

    # Hardware
    device: str = "auto"  # "auto", "cuda", or "cpu"
    use_amp: bool = True

    # Reproducibility
    seed: int = RANDOM_SEED

    # Paths (optional overrides)
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# =============================================================================
# Preset Configurations
# =============================================================================

def cpu_smoke_test_config() -> ExperimentConfig:
    """Minimal config for CPU testing (verify code works)."""
    return ExperimentConfig(
        name="cpu_smoke_test",
        description="Quick CPU test to verify code runs correctly",
        model_type="baseline",
        epochs=2,
        batch_size=2,
        num_workers=0,  # Avoid multiprocessing issues on Windows
        device="cpu",
        use_amp=False,  # AMP not supported on CPU
        image_size=256,  # Smaller images for speed
    )


def gpu_baseline_config() -> ExperimentConfig:
    """Full training config for baseline ResNet-50."""
    return ExperimentConfig(
        name="baseline_resnet50_full",
        description="Baseline ResNet-50 training on GPU (batch=16, epochs=20)",
        model_type="baseline",
        epochs=20,
        batch_size=16,
        num_workers=4,
        device="cuda",
        use_amp=True,
    )


def gpu_cbam_config() -> ExperimentConfig:
    """Full training config for CBAM-ResNet50."""
    return ExperimentConfig(
        name="cbam_resnet50_full",
        description="CBAM-ResNet50 training on GPU (batch=16, epochs=20)",
        model_type="cbam",
        epochs=20,
        batch_size=16,
        num_workers=4,
        device="cuda",
        use_amp=True,
    )


# =============================================================================
# Config Loading Utility
# =============================================================================

def load_config(config_path: Optional[str], cli_args) -> Any:
    """Load config from file and merge with CLI arguments.

    CLI arguments take precedence over config file values.
    """
    if config_path is None:
        return cli_args

    config = ExperimentConfig.from_yaml(config_path)

    # CLI overrides (only if explicitly set)
    if hasattr(cli_args, 'epochs') and cli_args.epochs != EPOCHS:
        config.epochs = cli_args.epochs
    if hasattr(cli_args, 'lr') and cli_args.lr != LEARNING_RATE:
        config.learning_rate = cli_args.lr
    if hasattr(cli_args, 'batch_size') and cli_args.batch_size != BATCH_SIZE:
        config.batch_size = cli_args.batch_size
    if hasattr(cli_args, 'fold'):
        config.fold = cli_args.fold
    if hasattr(cli_args, 'model') and cli_args.model:
        config.model_type = cli_args.model

    return config
```

**YAML Config Files** (`src/configs/`):

**File**: `src/configs/cpu_smoke_test.yaml`
```yaml
# CPU Smoke Test Configuration
# Use this to verify code works before running full GPU training
name: cpu_smoke_test
description: Quick CPU test to verify code runs correctly

# Model
model_type: baseline
pretrained: true
dropout_rate: 0.5

# Training (minimal for speed)
epochs: 2
learning_rate: 0.0001
weight_decay: 0.0001
batch_size: 2
grad_clip_norm: 1.0
early_stopping_patience: 5

# Loss
focal_gamma: 2.0
use_class_weights: true

# Data (reduced for CPU)
image_size: 256
num_workers: 0  # Avoid multiprocessing on Windows
fold: 0

# Hardware
device: cpu
use_amp: false

# Reproducibility
seed: 42
```

**File**: `src/configs/gpu_baseline.yaml`
```yaml
# GPU Baseline ResNet-50 Configuration
# Full training for proper baseline comparison
name: baseline_resnet50_full
description: Baseline ResNet-50 training on GPU

# Model
model_type: baseline
pretrained: true
dropout_rate: 0.5

# Training
epochs: 20
learning_rate: 0.0001
weight_decay: 0.0001
batch_size: 16
grad_clip_norm: 1.0
early_stopping_patience: 5

# Loss
focal_gamma: 2.0
use_class_weights: true

# Data
image_size: 512
num_workers: 4
fold: 0

# Hardware
device: cuda
use_amp: true

# Reproducibility
seed: 42
```

**File**: `src/configs/gpu_cbam.yaml`
```yaml
# GPU CBAM-ResNet50 Configuration
# Full training for the proposed model
name: cbam_resnet50_full
description: CBAM-ResNet50 training on GPU

# Model
model_type: cbam
pretrained: true
dropout_rate: 0.5

# Training
epochs: 20
learning_rate: 0.0001
weight_decay: 0.0001
batch_size: 16
grad_clip_norm: 1.0
early_stopping_patience: 5

# Loss
focal_gamma: 2.0
use_class_weights: true

# Data
image_size: 512
num_workers: 4
fold: 0

# Hardware
device: cuda
use_amp: true

# Reproducibility
seed: 42
```

---

## Phase 1: Baseline Retraining

### 1.1 CPU Smoke Test

**Purpose**: Verify all code changes from Phase 0 work correctly before committing to GPU training.

**Command**:
```bash
# Option 1: Using config file
python src/train.py --config src/configs/cpu_smoke_test.yaml --fold 0

# Option 2: Using CLI arguments
python src/train.py --model baseline --epochs 2 --batch_size 2 --fold 0
```

**Expected Output**:
```
Device: cpu
Loading APTOS 2019 data ...
  Train: 2929  |  Val: 733
Building BASELINE model ...
  Parameters: 23,512,389
  Class alpha weights: [0.406 2.121 1.091 3.763 6.315]

=================================================================
  BASELINE ResNet-50 - Fold 0
  Epochs: 1 -> 2  |  AMP: False  |  Device: cpu
=================================================================

Epoch 1/2 [Train]: 100%|██████████| 1464/1464 [time]
Epoch 1/2 [Val]: 100%|██████████| 367/367 [time]

  Epoch 1/2
    Train  - loss: X.XXXX  acc: 0.XXXX
    Val    - loss: X.XXXX  acc: 0.XXXX
    Val kappa: 0.XXXX  AUC: 0.XXXX  LR: 1.00e-04
  New best model saved (kappa = 0.XXXX)

... (epoch 2) ...

Training done - Best Val kappa: 0.XXXX
```

**Verification Checklist**:
- [ ] Training loop completes without errors
- [ ] Validation produces deterministic results (same kappa each validation call)
- [ ] Checkpoint saves successfully
- [ ] History JSON is written to `outputs/logs/`
- [ ] Progress bar shows correct epoch counts

### 1.2 GPU Full Training — ✅ COMPLETE (2026-03-31)

**Purpose**: Train baseline ResNet-50 with identical pipeline to CBAM model for valid ablation.

**Command Executed**:
```bash
python src/train.py --model baseline --epochs 20 --batch_size 16 --fold 0
```

**Training Infrastructure**:
- **GPU**: NVIDIA GeForce RTX 3090 (25.3 GB VRAM)
- **Platform**: Vast.ai cloud instance
- **Training Time**: ~2.5 hours (19 epochs completed)
- **Training Set**: 2,929 images
- **Validation Set**: 733 images

**Actual Outputs**:
```
dr-results/
├── outputs/
│   ├── checkpoints/
│   │   ├── baseline_resnet50_fold0_best.pth   # 270 MB (epoch 14)
│   │   └── baseline_resnet50_fold0_last.pth   # 270 MB (epoch 19)
│   ├── logs/
│   │   └── baseline_resnet50_fold0_history.json
│   ├── results/
│   │   └── baseline_resnet50_fold0_metrics.json
│   └── figures/                                # 19 visualizations
│       ├── baseline_resnet50_training_curves.png
│       ├── baseline_resnet50_confusion_matrix.png
│       ├── baseline_resnet50_per_class_recall.png
│       └── [16 additional dataset/domain analysis figures]
└── training_baseline_fold0.log                  # Full training log
```

**Actual Results** (exceeds expected performance):
| Metric              | Expected Range | **Actual (Best)** | **Actual (Final)** | Epoch |
| ------------------- | -------------- | ----------------- | ------------------ | ----- |
| **Val QWK**         | 0.75 - 0.82    | **0.9088**        | 0.9015             | 14    |
| **Val AUC**         | 0.85 - 0.92    | **0.9846**        | 0.9848             | 14    |
| **Val Acc**         | 0.72 - 0.78    | **0.8445**        | 0.8295             | 14    |
| **Train Acc**       | —              | 0.8982            | 0.9259             | 14/19 |
| **Train Loss**      | —              | 0.175             | 0.127              | 14/19 |
| **Val Loss**        | —              | 0.535             | 0.610              | 14/19 |

**Key Observations**:
1. ✅ **Outstanding performance**: Val QWK 0.9088 significantly exceeds expected baseline (0.75-0.82)
2. ✅ **Strong generalization**: 84.45% validation accuracy with stable training
3. ✅ **Early convergence**: Best model at epoch 14, training stopped at epoch 19
4. ✅ **Focal Loss effectiveness**: Successfully handled class imbalance (grade 0: 49% vs. grade 3: 5.3%)
5. ✅ **AMP stability**: Mixed precision training completed without gradient issues
6. ⚠️ **Overfitting signs**: Train accuracy (89.82%) > Val accuracy (84.45%) at epoch 14
7. ℹ️ **Early stopping**: Training halted before epoch 20 (likely early stopping or time limit)

**Visualizations Generated**: 19 publication-quality figures including training curves, confusion matrices, class distribution analysis, preprocessing comparisons, and domain shift characterization (APTOS vs. Messidor-2).

### 1.3 Messidor-2 Evaluation

**Purpose**: External validation to test domain generalization.

**Command**:
```bash
python src/evaluate.py \
    --checkpoint outputs/checkpoints/baseline_resnet50_fold0_best.pth \
    --batch_size 16 \
    --mc_passes 20
```

**Expected Output**:
```
Loading Messidor-2 dataset ...
  Images: 690
Loading checkpoint ...
  Checkpoint epoch: X  |  best kappa: 0.XXXX
Running MC Dropout inference (T = 20) ...
  Results CSV: outputs/results/messidor2_uncertainty.csv  (690 images)

==================================================
  MESSIDOR-2 EVALUATION
==================================================
  Accuracy:          0.XXXX
  Quadratic kappa:   0.XXXX    # Expect 5-15 points below APTOS val
  Referable DR AUC:  0.XXXX

==================================================
  UNCERTAINTY SUMMARY
==================================================
  Mean entropy:   0.XXXX
  Median entropy: 0.XXXX
  Max entropy:    X.XXXX
  Mean confidence: 0.XXXX

  Warning: X images (X.X%) above 90th-percentile entropy - consider manual review.
```

**Expected Files**:
```
outputs/
├── results/
│   ├── messidor2_uncertainty.csv
│   └── messidor2_metrics.json
└── figures/
    ├── messidor2_entropy_histogram.png
    └── messidor2_confidence_vs_entropy.png
```

---

## Directory Structure

After implementation:

```
src/
├── __init__.py
├── config.py                    # UPDATED: Added GRAD_CLIP_NORM
├── configs/                     # NEW: Experiment config system
│   ├── __init__.py
│   ├── experiment_config.py
│   ├── cpu_smoke_test.yaml
│   ├── gpu_baseline.yaml
│   └── gpu_cbam.yaml
├── model.py                     # UPDATED: MCDropout.mc_active, BaselineResNet50
├── train.py                     # UPDATED: --model, gradient clipping, config
├── dataset.py                   # (unchanged)
├── loss.py                      # (unchanged)
├── evaluate.py                  # (unchanged)
└── preprocessing.py             # (unchanged)
```

---

## Detailed TODO Checklist

### Phase 0: Infrastructure Fixes

| # | Task | Status | File(s) | Test |
|---|------|--------|---------|------|
| 0.1.1 | Add `mc_active` attribute to `MCDropout` | `[ ]` | `model.py` | Unit test |
| 0.1.2 | Modify `MCDropout.forward()` to check `mc_active` | `[ ]` | `model.py` | Unit test |
| 0.1.3 | Add `deterministic_mode()` context manager to `CBAMResNet50` | `[ ]` | `model.py` | Unit test |
| 0.1.4 | Update `Trainer.validate()` to use `deterministic_mode()` | `[ ]` | `train.py` | Integration |
| 0.1.5 | Write and run deterministic mode test script | `[ ]` | `tests/` | Manual |
| 0.2.1 | Add `GRAD_CLIP_NORM = 1.0` to config | `[ ]` | `config.py` | - |
| 0.2.2 | Add `grad_clip_norm` parameter to `Trainer.__init__` | `[ ]` | `train.py` | - |
| 0.2.3 | Add `scaler.unscale_()` + `clip_grad_norm_()` to `train_epoch()` | `[ ]` | `train.py` | Training run |
| 0.3.1 | Add `self.num_epochs` to `Trainer.__init__` | `[ ]` | `train.py` | - |
| 0.3.2 | Set `self.num_epochs = num_epochs` in `fit()` | `[ ]` | `train.py` | - |
| 0.3.3 | Update tqdm descriptions to use `self.num_epochs` | `[ ]` | `train.py` | Visual check |
| 0.4.1 | Create `BaselineResNet50` class | `[ ]` | `model.py` | Unit test |
| 0.4.2 | Add `deterministic_mode()` to `BaselineResNet50` | `[ ]` | `model.py` | Unit test |
| 0.4.3 | Create `create_baseline_model()` factory | `[ ]` | `model.py` | Import test |
| 0.5.1 | Add `--model` argument to argparse | `[ ]` | `train.py` | CLI test |
| 0.5.2 | Add model selection logic in `main()` | `[ ]` | `train.py` | Training run |
| 0.5.3 | Add `model_name` parameter to `Trainer` | `[ ]` | `train.py` | - |
| 0.5.4 | Update checkpoint/history paths to use `model_name` | `[ ]` | `train.py` | File check |
| 0.6.1 | Create `src/configs/` directory | `[ ]` | - | - |
| 0.6.2 | Implement `ExperimentConfig` dataclass | `[ ]` | `configs/experiment_config.py` | Import test |
| 0.6.3 | Create `cpu_smoke_test.yaml` | `[ ]` | `configs/` | - |
| 0.6.4 | Create `gpu_baseline.yaml` | `[ ]` | `configs/` | - |
| 0.6.5 | Create `gpu_cbam.yaml` | `[ ]` | `configs/` | - |
| 0.6.6 | Add `--config` CLI argument | `[ ]` | `train.py` | CLI test |
| 0.6.7 | Implement `load_config()` utility | `[ ]` | `train.py` or `configs/` | Integration |

### Phase 1: Baseline Retraining

| # | Task | Status | Depends On | Notes |
|---|------|--------|------------|-------|
| 1.1.1 | Run CPU smoke test (baseline) | `[ ]` | Phase 0 complete | 2 epochs, batch=2 |
| 1.1.2 | Verify deterministic validation | `[ ]` | 1.1.1 | Check val kappa reproducible |
| 1.1.3 | Verify checkpoint saved correctly | `[ ]` | 1.1.1 | Check file exists |
| 1.1.4 | Run CPU smoke test (CBAM) | `[ ]` | 1.1.2 | Verify CBAM also works |
| 1.2.1 | Run GPU baseline training (fold 0) | `[ ]` | 1.1.x | 20 epochs, batch=16 |
| 1.2.2 | Record val metrics (QWK, AUC, Acc) | `[ ]` | 1.2.1 | From final epoch |
| 1.2.3 | Save training history plot | `[ ]` | 1.2.1 | Loss + kappa curves |
| 1.3.1 | Run Messidor-2 evaluation | `[ ]` | 1.2.1 | MC Dropout T=20 |
| 1.3.2 | Record external metrics | `[ ]` | 1.3.1 | QWK, AUC, entropy stats |
| 1.3.3 | Save uncertainty figures | `[ ]` | 1.3.1 | Histogram, scatter |
| 1.3.4 | Analyze domain shift magnitude | `[ ]` | 1.3.2 | Compare to APTOS val |

---

## Verification Criteria

### Phase 0 Success Criteria

1. **MCDropout Deterministic Mode**
   - Running validation twice on the same data produces identical kappa values
   - MC Dropout inference (T=20) still produces different outputs per pass
   - Test script passes without assertion errors

2. **Gradient Clipping**
   - Training completes without NaN losses
   - No exploding gradient warnings
   - Gradient norms logged are <= 1.0 (if we add logging)

3. **Progress Bar Fix**
   - `--epochs 30` shows "Epoch X/30" in progress bar
   - Default (no argument) shows "Epoch X/20"

4. **Baseline Model**
   - `create_baseline_model()` returns a model
   - Model output shape is (B, 5)
   - Parameter count ~23.5M (same as ResNet-50)
   - `deterministic_mode()` works correctly

5. **Config System**
   - `--config cpu_smoke_test.yaml` runs on CPU
   - `--config gpu_baseline.yaml` runs on GPU
   - CLI arguments override config file values

### Phase 1 Success Criteria

1. **CPU Smoke Test**
   - Both baseline and CBAM models train for 2 epochs without errors
   - Checkpoints are saved with correct naming convention
   - Validation is deterministic (same results each call)

2. **GPU Baseline Training**
   - Training completes all 20 epochs (or early stops)
   - Best val QWK >= 0.75 (sanity check)
   - Checkpoint file size ~100MB

3. **Messidor-2 Evaluation**
   - All 690 images processed
   - QWK computed (expect 0.55-0.75 due to domain shift)
   - Uncertainty figures generated
   - No file loading errors

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Deterministic mode breaks MC inference | Low | High | Add explicit unit test for both modes |
| Gradient clipping hurts convergence | Low | Medium | Keep max_norm=1.0 (standard value); monitor loss curves |
| Config system adds complexity | Medium | Low | Provide clear defaults; CLI always works |
| Baseline underperforms expected range | Medium | Medium | Check preprocessing; compare to notebook results |
| Messidor-2 path resolution fails | Low | Medium | Test with single image first; check extensions |

---

## Appendix: Quick Reference Commands

```bash
# Phase 0 verification
python -c "from src.model import create_model, create_baseline_model; print('Imports OK')"

# CPU smoke test
python src/train.py --model baseline --epochs 2 --batch_size 2 --fold 0

# GPU baseline full training
python src/train.py --config src/configs/gpu_baseline.yaml --fold 0

# Messidor-2 evaluation
python src/evaluate.py --checkpoint outputs/checkpoints/baseline_resnet50_fold0_best.pth

# View training history
cat outputs/logs/baseline_resnet50_fold0_history.json | python -m json.tool

# Check checkpoint
python -c "import torch; c = torch.load('outputs/checkpoints/baseline_resnet50_fold0_best.pth', map_location='cpu'); print(f'Epoch: {c[\"epoch\"]}, Best kappa: {c[\"best_kappa\"]:.4f}')"
```

---

## Master TODO List (CPU-First Workflow)

> **Priority Order**: Complete all tasks in sequence. CPU testing validates code before GPU investment.
> **Convention**: `[ ]` = pending, `[x]` = done, `[~]` = in progress, `[-]` = skipped/deferred

---

### PHASE 0: INFRASTRUCTURE FIXES

#### 0.1 — Fix MCDropout Deterministic Validation
> **Goal**: Eliminate stochastic noise during training validation while preserving MC Dropout for inference.

| ID | Task | File | Status | Depends |
|----|------|------|--------|---------|
| 0.1.1 | Read current `MCDropout` class implementation | `src/model.py` | `[x]` | — |
| 0.1.2 | Add `self.mc_active = True` attribute to `MCDropout.__init__()` | `src/model.py` | `[x]` | 0.1.1 |
| 0.1.3 | Modify `MCDropout.forward()` to check `if self.mc_active:` before applying dropout | `src/model.py` | `[x]` | 0.1.2 |
| 0.1.4 | Add `from contextlib import contextmanager` import at top of file | `src/model.py` | `[x]` | — |
| 0.1.5 | Implement `deterministic_mode()` context manager in `CBAMResNet50` class | `src/model.py` | `[x]` | 0.1.3, 0.1.4 |
| 0.1.6 | Write unit test for MCDropout deterministic mode | `src/tests/test_model.py` | `[x]` | 0.1.5 |
| 0.1.7 | Run unit test — verify outputs differ in MC mode, identical in deterministic mode | — | `[x]` | 0.1.6 |

**Test Script** (save as `src/tests/test_model.py`):
```python
"""Unit tests for model components."""
import torch
from model import MCDropout, CBAMResNet50, create_model

def test_mcdropout_deterministic_mode():
    """Verify deterministic mode produces identical outputs."""
    model = create_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)  # Smaller for CPU speed

    # MC mode (default): outputs should differ
    with torch.no_grad():
        o1 = model(x)
        o2 = model(x)
    assert not torch.allclose(o1, o2), "MC mode should produce different outputs"

    # Deterministic mode: outputs should be identical
    with torch.no_grad():
        with model.deterministic_mode():
            d1 = model(x)
            d2 = model(x)
    assert torch.allclose(d1, d2), "Deterministic mode should produce identical outputs"

    # After context exit, MC mode should be restored
    with torch.no_grad():
        o3 = model(x)
        o4 = model(x)
    assert not torch.allclose(o3, o4), "MC mode should be restored after context exit"

    print("✓ MCDropout deterministic mode test passed!")

if __name__ == "__main__":
    test_mcdropout_deterministic_mode()
```

---

#### 0.2 — Add Gradient Clipping
> **Goal**: Prevent exploding gradients from Focal Loss on mislabeled samples.

| ID | Task | File | Status | Depends |
|----|------|------|--------|---------|
| 0.2.1 | Add `GRAD_CLIP_NORM = 1.0` constant to config | `src/config.py` | `[x]` | — |
| 0.2.2 | Import `GRAD_CLIP_NORM` in train.py | `src/train.py` | `[x]` | 0.2.1 |
| 0.2.3 | Add `self.grad_clip_norm` parameter to `Trainer.__init__()` | `src/train.py` | `[x]` | 0.2.2 |
| 0.2.4 | Add `self.scaler.unscale_(optimizer)` before step in `train_epoch()` | `src/train.py` | `[x]` | 0.2.3 |
| 0.2.5 | Add `torch.nn.utils.clip_grad_norm_()` call after unscale | `src/train.py` | `[x]` | 0.2.4 |
| 0.2.6 | Test training runs without NaN loss (visual check during smoke test) | — | `[x]` | 0.2.5 |

---

#### 0.3 — Fix Progress Bar Epoch Display
> **Goal**: Show actual `--epochs` value instead of hardcoded constant.

| ID | Task | File | Status | Depends |
|----|------|------|--------|---------|
| 0.3.1 | Add `self.num_epochs = EPOCHS` in `Trainer.__init__()` | `src/train.py` | `[x]` | — |
| 0.3.2 | Set `self.num_epochs = num_epochs` at start of `fit()` method | `src/train.py` | `[x]` | 0.3.1 |
| 0.3.3 | Replace `EPOCHS` with `self.num_epochs` in `train_epoch()` tqdm desc | `src/train.py` | `[x]` | 0.3.2 |
| 0.3.4 | Replace `EPOCHS` with `self.num_epochs` in `validate()` tqdm desc | `src/train.py` | `[x]` | 0.3.2 |
| 0.3.5 | Visual verification: run `--epochs 3` and check progress bar shows `/3` | — | `[x]` | 0.3.4 |

---

#### 0.4 — Add Baseline Model Factory
> **Goal**: Create baseline ResNet-50 with identical head for fair ablation.

| ID | Task | File | Status | Depends |
|----|------|------|--------|---------|
| 0.4.1 | Design `BaselineResNet50` class structure (same head as CBAM) | — | `[x]` | — |
| 0.4.2 | Implement `BaselineResNet50.__init__()` with stem + layer1-4 | `src/model.py` | `[x]` | 0.4.1 |
| 0.4.3 | Add avgpool + mc_dropout + fc to `BaselineResNet50` (same as CBAM) | `src/model.py` | `[x]` | 0.4.2 |
| 0.4.4 | Implement `BaselineResNet50.forward()` | `src/model.py` | `[x]` | 0.4.3 |
| 0.4.5 | Add `deterministic_mode()` context manager to `BaselineResNet50` | `src/model.py` | `[x]` | 0.4.4, 0.1.5 |
| 0.4.6 | Create `create_baseline_model()` factory function | `src/model.py` | `[x]` | 0.4.5 |
| 0.4.7 | Write unit test for baseline model output shape | `src/tests/test_model.py` | `[x]` | 0.4.6 |
| 0.4.8 | Verify baseline param count (~23.5M) matches standard ResNet-50 | — | `[x]` | 0.4.7 |

**Test Script Addition**:
```python
def test_baseline_model():
    """Verify baseline model architecture."""
    from model import create_baseline_model

    model = create_baseline_model(pretrained=False)
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"

    param_count = sum(p.numel() for p in model.parameters())
    assert 23_000_000 < param_count < 24_000_000, f"Unexpected param count: {param_count:,}"

    print(f"✓ Baseline model test passed! Params: {param_count:,}")
```

---

#### 0.5 — Add --model CLI Argument
> **Goal**: Support both baseline and CBAM training with same pipeline.

| ID | Task | File | Status | Depends |
|----|------|------|--------|---------|
| 0.5.1 | Add `--model` argument to argparse (choices: baseline, cbam) | `src/train.py` | `[x]` | — |
| 0.5.2 | Import `create_baseline_model` from model.py | `src/train.py` | `[x]` | 0.4.6 |
| 0.5.3 | Add if/else block in `main()` to select model based on `args.model` | `src/train.py` | `[x]` | 0.5.1, 0.5.2 |
| 0.5.4 | Create `model_name` variable based on model type | `src/train.py` | `[x]` | 0.5.3 |
| 0.5.5 | Add `model_name` parameter to `Trainer.__init__()` | `src/train.py` | `[x]` | 0.5.4 |
| 0.5.6 | Update `save_checkpoint()` to use `self.model_name` in filename | `src/train.py` | `[x]` | 0.5.5 |
| 0.5.7 | Update `_save_history()` to use `self.model_name` in filename | `src/train.py` | `[x]` | 0.5.5 |
| 0.5.8 | Test: `--model baseline` creates `baseline_resnet50_fold0_*.pth` | — | `[x]` | 0.5.7 |
| 0.5.9 | Test: `--model cbam` creates `cbam_resnet50_fold0_*.pth` | — | `[x]` | 0.5.7 |

---

#### 0.6 — Create Experiment Config System
> **Goal**: YAML-based configs for CPU smoke tests and GPU full training.

| ID | Task | File | Status | Depends |
|----|------|------|--------|---------|
| 0.6.1 | Create `src/configs/` directory | — | `[x]` | — |
| 0.6.2 | Create `src/configs/__init__.py` (empty) | `src/configs/__init__.py` | `[x]` | 0.6.1 |
| 0.6.3 | Implement `ExperimentConfig` dataclass | `src/configs/experiment_config.py` | `[x]` | 0.6.2 |
| 0.6.4 | Add `from_yaml()` classmethod to ExperimentConfig | `src/configs/experiment_config.py` | `[x]` | 0.6.3 |
| 0.6.5 | Add `to_yaml()` method to ExperimentConfig | `src/configs/experiment_config.py` | `[x]` | 0.6.3 |
| 0.6.6 | Add `get_device()` method (resolves "auto" to actual device) | `src/configs/experiment_config.py` | `[x]` | 0.6.3 |
| 0.6.7 | Create `cpu_smoke_test.yaml` config file | `src/configs/cpu_smoke_test.yaml` | `[x]` | 0.6.3 |
| 0.6.8 | Create `gpu_baseline.yaml` config file | `src/configs/gpu_baseline.yaml` | `[x]` | 0.6.3 |
| 0.6.9 | Create `gpu_cbam.yaml` config file | `src/configs/gpu_cbam.yaml` | `[x]` | 0.6.3 |
| 0.6.10 | Add `--config` argument to argparse in train.py | `src/train.py` | `[x]` | — |
| 0.6.11 | Implement `load_config()` utility function | `src/configs/experiment_config.py` | `[x]` | 0.6.4 |
| 0.6.12 | Integrate config loading into `main()` | `src/train.py` | `[x]` | 0.6.10, 0.6.11 |
| 0.6.13 | Test: `--config cpu_smoke_test.yaml` loads correct settings | — | `[x]` | 0.6.12 |

---

### PHASE 1: BASELINE RETRAINING (CPU-FIRST)

#### 1.1 — CPU Smoke Test (Baseline Model)
> **Goal**: Verify all Phase 0 changes work correctly on CPU before GPU investment.

| ID | Task | File | Status | Depends |
|----|------|------|--------|---------|
| 1.1.1 | Run unit tests: `python src/tests/test_model.py` | — | `[x]` | Phase 0 |
| 1.1.2 | Run CPU smoke test: `python src/train.py --model baseline --epochs 2 --batch_size 2` | — | `[x]` | 1.1.1 |
| 1.1.3 | Verify training loop completes both epochs without error | — | `[x]` | 1.1.2 |
| 1.1.4 | Verify progress bar shows "Epoch X/2" (not hardcoded /20) | — | `[x]` | 1.1.2 |
| 1.1.5 | Verify checkpoint file created: `outputs/checkpoints/baseline_resnet50_fold0_best.pth` | — | `[x]` | 1.1.2 |
| 1.1.6 | Verify history JSON created: `outputs/logs/baseline_resnet50_fold0_history.json` | — | `[x]` | 1.1.2 |
| 1.1.7 | Run validation twice manually — verify kappa is identical (deterministic) | — | `[x]` | 1.1.2 |
| 1.1.8 | Check no NaN values in loss output (gradient clipping working) | — | `[x]` | 1.1.2 |

**Verification Commands**:
```bash
# Run smoke test
python src/train.py --model baseline --epochs 2 --batch_size 2 --fold 0

# Check checkpoint exists
ls -la outputs/checkpoints/baseline_resnet50_fold0_*.pth

# Inspect checkpoint
python -c "import torch; c = torch.load('outputs/checkpoints/baseline_resnet50_fold0_best.pth', map_location='cpu'); print(f'Epoch: {c[\"epoch\"]}, Kappa: {c[\"best_kappa\"]:.4f}')"

# Check history
cat outputs/logs/baseline_resnet50_fold0_history.json
```

---

#### 1.2 — CPU Smoke Test (CBAM Model)
> **Goal**: Verify CBAM model also works with the new infrastructure.

| ID | Task | File | Status | Depends |
|----|------|------|--------|---------|
| 1.2.1 | Run CPU smoke test: `python src/train.py --model cbam --epochs 2 --batch_size 2` | — | `[x]` | 1.1.8 |
| 1.2.2 | Verify training completes without error | — | `[x]` | 1.2.1 |
| 1.2.3 | Verify checkpoint: `outputs/checkpoints/cbam_resnet50_fold0_best.pth` | — | `[x]` | 1.2.1 |
| 1.2.4 | Verify history: `outputs/logs/cbam_resnet50_fold0_history.json` | — | `[x]` | 1.2.1 |
| 1.2.5 | Compare param count: CBAM (~25.5M) vs baseline (~23.5M) — CBAM should be higher | — | `[x]` | 1.2.1 |

---

#### 1.3 — CPU Validation Determinism Test
> **Goal**: Confirm validation produces reproducible metrics.

| ID | Task | File | Status | Depends |
|----|------|------|--------|---------|
| 1.3.1 | Create test script that runs validation 3 times on same loader | `src/tests/test_determinism.py` | `[x]` | 1.1.7 |
| 1.3.2 | Run test script — all 3 validation kappas must be identical | — | `[x]` | 1.3.1 |
| 1.3.3 | Document validation kappa values in test output | — | `[x]` | 1.3.2 |

**Test Script** (save as `src/tests/test_determinism.py`):
```python
"""Test validation determinism with new MCDropout changes."""
import torch
import sys
sys.path.append('src')

from model import create_model, create_baseline_model
from dataset import get_val_transform, DRDataset, get_train_val_split
from torch.utils.data import DataLoader
from config import APTOS_TRAIN_CSV, APTOS_TRAIN_IMAGES
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

def run_validation(model, loader):
    """Run single validation pass and return kappa."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        with model.deterministic_mode():
            for images, labels in loader:
                logits = model(images)
                preds = logits.argmax(dim=1).numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return kappa

def test_validation_determinism():
    """Run validation 3 times, verify identical results."""
    # Load small subset for speed
    df = pd.read_csv(APTOS_TRAIN_CSV)
    _, val_df = get_train_val_split(df, val_fold=0)
    val_df = val_df.head(100)  # Small subset

    val_dataset = DRDataset(
        df=val_df,
        image_dir=APTOS_TRAIN_IMAGES,
        transform=get_val_transform(256),
        target_size=256,
    )
    loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = create_baseline_model(pretrained=False)

    kappas = []
    for i in range(3):
        k = run_validation(model, loader)
        kappas.append(k)
        print(f"  Run {i+1}: kappa = {k:.6f}")

    assert kappas[0] == kappas[1] == kappas[2], \
        f"Validation not deterministic! Kappas: {kappas}"

    print("✓ Validation determinism test passed!")

if __name__ == "__main__":
    test_validation_determinism()
```

---

#### 1.4 — GPU Full Training (Baseline) — DEFERRED
> **Status**: Deferred until CPU smoke tests pass.

| ID | Task | Status | Notes |
|----|------|--------|-------|
| 1.4.1 | Run: `python src/train.py --config src/configs/gpu_baseline.yaml` | `[-]` | Deferred |
| 1.4.2 | Monitor training for 20 epochs (~45-60 min on RTX 3080) | `[-]` | Deferred |
| 1.4.3 | Record final val QWK (expect 0.75-0.82) | `[-]` | Deferred |
| 1.4.4 | Record final val AUC (expect 0.85-0.92) | `[-]` | Deferred |
| 1.4.5 | Save training curves (loss + kappa) | `[-]` | Deferred |

---

#### 1.5 — Messidor-2 Evaluation — DEFERRED
> **Status**: Deferred until GPU training complete.

| ID | Task | Status | Notes |
|----|------|--------|-------|
| 1.5.1 | Run baseline evaluation on Messidor-2 | `[-]` | Deferred |
| 1.5.2 | Record external QWK (expect 5-15 points below APTOS) | `[-]` | Deferred |
| 1.5.3 | Generate uncertainty figures | `[-]` | Deferred |

---

### SUMMARY: Current Sprint Tasks (CPU-First)

**STATUS: ALL CPU TASKS COMPLETED**

```
Phase 0 Tasks: 42 total - ALL COMPLETE [x]
├── 0.1 MCDropout Fix:      7 tasks [x]
├── 0.2 Gradient Clipping:  6 tasks [x]
├── 0.3 Progress Bar:       5 tasks [x]
├── 0.4 Baseline Model:     8 tasks [x]
├── 0.5 CLI Arguments:      9 tasks [x]
└── 0.6 Config System:     13 tasks [x]

Phase 1 CPU Tasks: 16 total - ALL COMPLETE [x]
├── 1.1 Baseline Smoke:     8 tasks [x]
├── 1.2 CBAM Smoke:         5 tasks [x]
└── 1.3 Determinism Test:   3 tasks [x]

GPU Tasks: DEFERRED (for when GPU available)
├── 1.4 GPU Training:       5 tasks [-]
└── 1.5 Messidor-2:         3 tasks [-]
```

---

### Implementation Order (Recommended)

```
START
  │
  ▼
┌─────────────────────────────────────────┐
│ 0.1 Fix MCDropout (model.py)            │ ← First priority
│   - mc_active toggle                    │
│   - deterministic_mode() context        │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ 0.4 Add BaselineResNet50 (model.py)     │ ← Depends on 0.1
│   - Same head as CBAM                   │
│   - Include deterministic_mode()        │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ 0.2 + 0.3 Training fixes (train.py)     │
│   - Gradient clipping                   │
│   - Progress bar fix                    │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ 0.5 CLI --model argument (train.py)     │ ← Depends on 0.4
│   - model selection                     │
│   - dynamic checkpoint naming           │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ 0.6 Config system (configs/)            │ ← Can be parallel
│   - ExperimentConfig dataclass          │
│   - YAML presets                        │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ 1.1-1.3 CPU Smoke Tests                 │
│   - Baseline + CBAM                     │
│   - Determinism verification            │
└─────────────────────────────────────────┘
  │
  ▼
 END CPU PHASE
  │
  ▼ (when GPU available)
┌─────────────────────────────────────────┐
│ 1.4-1.5 GPU Training + Evaluation       │
│   - Full 20-epoch training              │
│   - Messidor-2 external validation      │
└─────────────────────────────────────────┘
```

---

## Next Steps After Phase 1

After completing Phase 0 and Phase 1, proceed to:

1. **Phase 2**: Train CBAM-ResNet50 (fold 0) using `gpu_cbam.yaml`
2. **Phase 3**: Implement ECE computation and referral threshold analysis
3. **Phase 4**: Run ablation study (4 model variants)

The infrastructure fixes from Phase 0 ensure all subsequent experiments share the same training pipeline, enabling valid ablation comparisons.
