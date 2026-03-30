"""
Training script for the Uncertainty-Aware Attention CNN.

Key features
------------
- **Mixed-precision training** (``torch.amp``): halves VRAM for 512×512
  inputs, enabling batch_size=16 on a single 12 GB GPU.
- **Focal Loss** with per-class alpha weights computed from the training
  set class frequencies — handles the severe APTOS 2019 class imbalance.
- **AdamW + CosineAnnealingLR**: decoupled weight decay + smooth LR decay.
- **Full reproducibility**: ``seed_everything()`` fixes Python, NumPy,
  PyTorch, and CuDNN random states.

Usage::

    python train.py                         # defaults
    python train.py --epochs 30 --lr 3e-4   # overrides
    python train.py --resume outputs/checkpoints/best.pth
"""

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score

from config import (
    APTOS_TRAIN_CSV, APTOS_TRAIN_IMAGES,
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, SCHEDULER_T_MAX,
    EARLY_STOPPING_PATIENCE, CHECKPOINT_DIR, LOG_DIR,
    BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE, RANDOM_SEED,
    MC_DROPOUT_RATE, NUM_CLASSES, FOCAL_GAMMA, USE_AMP,
    GRAD_CLIP_NORM, APTOS_PROCESSED_DIR, USE_PREPROCESSED_CACHE,
    seed_everything, setup_directories,
)
from model import create_model, create_baseline_model
from dataset import get_train_val_split, create_dataloaders, DRDataset
from loss import FocalLoss, compute_class_weights


# =========================================================================
#  Trainer
# =========================================================================

class Trainer:
    """End-to-end training manager with AMP, early stopping, and checkpointing.

    MLOps rationale
    ---------------
    * ``GradScaler`` + ``autocast`` are the standard AMP recipe.  They
      keep activations in float16 during the forward/backward pass and
      only do the optimizer step in float32, cutting peak VRAM by ~50 %.
    * ``zero_grad(set_to_none=True)`` avoids a memset to zero — saves one
      GPU kernel launch per step.
    * ``pin_memory=True`` in DataLoaders enables async CPU → GPU transfers,
      hiding data-loading latency behind compute.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        fold: int = 0,
        model_name: str = "cbam_resnet50",
        grad_clip_norm: float = GRAD_CLIP_NORM,
    ):
        self.model = model.to(device)
        self.device = device
        self.fold = fold
        self.model_name = model_name
        self.grad_clip_norm = grad_clip_norm

        # AMP scaler (no-op on CPU; active on CUDA when USE_AMP=True)
        self.scaler = torch.amp.GradScaler(
            device="cuda", enabled=(USE_AMP and device.type == "cuda")
        )
        self.amp_enabled = USE_AMP and device.type == "cuda"

        # Training state
        self.current_epoch = 0
        self.best_kappa = -1.0
        self.epochs_no_improve = 0
        self.num_epochs = EPOCHS

        # Metrics history
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
            "val_kappa": [],  "val_auc": [],
        }

    # -----------------------------------------------------------------
    #  Single epoch
    # -----------------------------------------------------------------

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
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_norm
            )
            self.scaler.step(optimizer)
            self.scaler.update()

            # --- Bookkeeping (on CPU, outside autocast) ---
            running_loss += loss.item() * images.size(0)
            preds = logits.detach().argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    # -----------------------------------------------------------------
    #  Validation
    # -----------------------------------------------------------------

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

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_kappa = cohen_kappa_score(
            all_labels, all_preds, weights="quadratic"
        )

        # Binary referable DR AUC  (grade >= 2 vs < 2)
        all_probs = np.vstack(all_probs)
        binary_labels = (np.array(all_labels) >= 2).astype(int)
        binary_probs = all_probs[:, 2:].sum(axis=1)
        try:
            epoch_auc = roc_auc_score(binary_labels, binary_probs)
        except ValueError:
            epoch_auc = 0.0

        return epoch_loss, epoch_acc, epoch_kappa, epoch_auc

    # -----------------------------------------------------------------
    #  Checkpointing
    # -----------------------------------------------------------------

    def save_checkpoint(self, optimizer, scheduler, is_best=False):
        """Save model + optimizer + scheduler for resume."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_kappa": self.best_kappa,
            "history": self.history,
            "model_name": self.model_name,
        }

        last_path = CHECKPOINT_DIR / f"{self.model_name}_fold{self.fold}_last.pth"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = CHECKPOINT_DIR / f"{self.model_name}_fold{self.fold}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"  New best model saved (kappa = {self.best_kappa:.4f})")

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
        """Resume from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.current_epoch = ckpt["epoch"] + 1           # resume from *next* epoch
        self.best_kappa = ckpt["best_kappa"]
        self.history = ckpt["history"]

        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        print(f"  Resumed from epoch {ckpt['epoch']} "
              f"(best κ = {self.best_kappa:.4f})")

    # -----------------------------------------------------------------
    #  Full training loop
    # -----------------------------------------------------------------

    def fit(self, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=EPOCHS):
        """Full training loop with early stopping."""
        self.num_epochs = num_epochs
        start_epoch = self.current_epoch

        print(f"\n{'=' * 65}")
        print(f"  {self.model_name} - Fold {self.fold}")
        print(f"  Epochs: {start_epoch + 1} -> {num_epochs}  |  "
              f"AMP: {self.amp_enabled}  |  "
              f"Device: {self.device}")
        print(f"{'=' * 65}\n")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            t_loss, t_acc = self.train_epoch(train_loader, criterion, optimizer)

            # Validate
            v_loss, v_acc, v_kappa, v_auc = self.validate(val_loader, criterion)

            # LR step
            scheduler.step()

            # Log
            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)
            self.history["val_kappa"].append(v_kappa)
            self.history["val_auc"].append(v_auc)

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"\n  Epoch {epoch + 1}/{num_epochs}\n"
                f"    Train  — loss: {t_loss:.4f}  acc: {t_acc:.4f}\n"
                f"    Val    — loss: {v_loss:.4f}  acc: {v_acc:.4f}\n"
                f"    Val κ: {v_kappa:.4f}  AUC: {v_auc:.4f}  LR: {lr:.2e}"
            )

            # Best-model tracking
            is_best = v_kappa > self.best_kappa
            if is_best:
                self.best_kappa = v_kappa
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(optimizer, scheduler, is_best)

            # Early stopping
            if self.epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        print(f"\n{'=' * 65}")
        print(f"  Training done - Best Val kappa: {self.best_kappa:.4f}")
        print(f"{'=' * 65}\n")

        self._save_history()

    def _save_history(self):
        """Persist training curves as JSON for later plotting."""
        path = LOG_DIR / f"{self.model_name}_fold{self.fold}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  History saved to {path}")


# =========================================================================
#  Entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train DR detection model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fold", type=int, default=0, help="Validation fold (0-4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument(
        "--model",
        type=str,
        default="cbam",
        choices=["baseline", "cbam"],
        help="Model architecture: 'baseline' (ResNet-50) or 'cbam' (CBAM-ResNet50)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides CLI args)"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=USE_PREPROCESSED_CACHE,
        help="Load preprocessed images from data/processed/ cache"
    )
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        try:
            from configs.experiment_config import load_config
            args = load_config(args.config, args)
        except ImportError:
            print("Warning: Config system not available, using CLI args")

    # ---- Reproducibility ----
    seed_everything(RANDOM_SEED)
    setup_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Data ----
    print("\nLoading APTOS 2019 data ...")
    df = pd.read_csv(APTOS_TRAIN_CSV)
    train_df, val_df = get_train_val_split(df, val_fold=args.fold)
    print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}")

    use_cache = args.use_cache and APTOS_PROCESSED_DIR.exists()
    cache_dir = APTOS_PROCESSED_DIR if use_cache else None
    if use_cache:
        print(f"  Using preprocessed cache: {APTOS_PROCESSED_DIR}")

    train_loader, val_loader = create_dataloaders(
        train_df, val_df, APTOS_TRAIN_IMAGES,
        batch_size=args.batch_size, num_workers=NUM_WORKERS,
        use_cache=use_cache, cache_dir=cache_dir,
    )

    # ---- Model Selection ----
    print(f"\nBuilding {args.model.upper()} model ...")
    if args.model == "baseline":
        model = create_baseline_model(
            num_classes=NUM_CLASSES,
            dropout_rate=MC_DROPOUT_RATE,
            pretrained=True,
        )
        model_name = "baseline_resnet50"
    else:
        model = create_model(
            num_classes=NUM_CLASSES,
            dropout_rate=MC_DROPOUT_RATE,
            pretrained=True,
        )
        model_name = "cbam_resnet50"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ---- Loss (Focal) ----
    train_labels = torch.tensor(train_df["diagnosis"].values)
    alpha_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
    print(f"  Class alpha weights: {alpha_weights.cpu().numpy().round(3)}")

    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha_weights)

    # ---- Optimizer + Scheduler ----
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---- Trainer ----
    trainer = Trainer(
        model, device,
        fold=args.fold,
        model_name=model_name,
        grad_clip_norm=GRAD_CLIP_NORM,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume, optimizer, scheduler)

    trainer.fit(
        train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=args.epochs,
    )

    print("\nTraining finished!")


if __name__ == "__main__":
    main()
