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
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, f1_score,
)

from config import (
    APTOS_TRAIN_CSV, APTOS_TRAIN_IMAGES,
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, SCHEDULER_T_MAX,
    EARLY_STOPPING_PATIENCE, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR,
    BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE, RANDOM_SEED, N_FOLDS,
    MC_DROPOUT_RATE, NUM_CLASSES, FOCAL_GAMMA, USE_AMP,
    GRAD_CLIP_NORM, APTOS_PROCESSED_DIR, USE_PREPROCESSED_CACHE,
    LABEL_SMOOTHING, USE_BALANCED_SAMPLER, CLASSIFIER_HIDDEN_DIM,
    LR_WARMUP_EPOCHS,
    seed_everything, setup_directories,
)
from model import create_model, create_baseline_model
from dataset import get_train_val_split, create_dataloaders, DRDataset
from loss import FocalLoss, compute_class_weights


#  Trainer


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        fold: int = 0,
        model_name: str = "cbam_resnet50",
        grad_clip_norm: float = GRAD_CLIP_NORM,
        hyperparams: dict | None = None,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    ):
        self.model = model.to(device)
        self.device = device
        self.fold = fold
        self.model_name = model_name
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_patience = early_stopping_patience

        # AMP scaler chi dung tren GPU, dung de tu dong scale loss va unscale gradients
        self.scaler = torch.amp.GradScaler(
            device="cuda", enabled=(USE_AMP and device.type == "cuda")
        )
        self.amp_enabled = USE_AMP and device.type == "cuda"

        # Training state
        self.current_epoch = 0
        self.best_kappa = -1.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.num_epochs = EPOCHS

        # Timestamp
        from datetime import datetime
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_tag = f"{model_name}_{self.run_timestamp}_fold{fold}"
        self.start_time = None  # Set when fit() starts
        self.end_time = None    # Set when fit() ends

        self.hyperparams = hyperparams or {}

        # METRICS HISTORY 
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
            "val_kappa": [],  "val_auc": [],
            "val_sens": [],   "val_spec": [],  # Binary referable DR metrics
            "val_f1_macro": [],  # Macro F1 across all 5 classes
            "val_recall_per_class": [],  # Per-class recal
            "val_precision_per_class": [],  # Per-class precision
            "val_f1_per_class": [],  # Per-class F1-score
        }

  

    def train_epoch(self, train_loader, criterion, optimizer):
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

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.amp_enabled
            ):
                logits = self.model(images)
                loss = criterion(logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_norm
            )


            self.scaler.step(optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = logits.detach().argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    # Val

    @torch.no_grad()
    def validate(self, val_loader, criterion):
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

        # Binary referable DR metrics (grade >= 2 vs < 2)
        all_probs = np.vstack(all_probs)
        binary_labels = (np.array(all_labels) >= 2).astype(int)
        binary_preds = (np.array(all_preds) >= 2).astype(int)
        binary_probs = all_probs[:, 2:].sum(axis=1)

        # AUC
        try:
            epoch_auc = roc_auc_score(binary_labels, binary_probs)
        except ValueError:
            epoch_auc = 0.0

        # Sensitivity & Specificity 
        tn, fp, fn, tp = confusion_matrix(
            binary_labels, binary_preds, labels=[0, 1]
        ).ravel()
        epoch_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        epoch_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # PEr clsss
        report = classification_report(
            all_labels, all_preds,
            labels=[0, 1, 2, 3, 4],
            output_dict=True,
            zero_division=0,
        )
        per_class = {
            "recall":    [report[str(c)]["recall"]    for c in range(5)],
            "precision": [report[str(c)]["precision"] for c in range(5)],
            "f1":        [report[str(c)]["f1-score"]  for c in range(5)],
            "macro_f1":  report["macro avg"]["f1-score"],
        }

        return (epoch_loss, epoch_acc, epoch_kappa, epoch_auc,
                epoch_sens, epoch_spec, per_class)

    #  Checkpoint

    def save_checkpoint(self, optimizer, scheduler, is_best=False):
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_kappa": self.best_kappa,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "model_name": self.model_name,
            "run_timestamp": self.run_timestamp,
            "run_tag": self.run_tag,
            "hyperparams": self.hyperparams,
        }

        last_path = CHECKPOINT_DIR / f"{self.run_tag}_last.pth"
        torch.save(checkpoint, last_path)

        if is_best:
            self.best_epoch = self.current_epoch
            best_path = CHECKPOINT_DIR / f"{self.run_tag}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"  New best model saved (kappa = {self.best_kappa:.4f})")

    def load_checkpoint(self, path, optimizer=None, scheduler=None):
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
        import time
        from datetime import datetime

        self.num_epochs = num_epochs
        start_epoch = self.current_epoch
        self.start_time = time.time()

        print(f"\n{'=' * 65}")
        print(f"  {self.model_name} - Fold {self.fold}")
        print(f"  Run Tag: {self.run_tag}")
        print(f"  Epochs: {start_epoch + 1} -> {num_epochs}  |  "
              f"AMP: {self.amp_enabled}  |  "
              f"Device: {self.device}")
        print(f"{'=' * 65}\n")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            t_loss, t_acc = self.train_epoch(train_loader, criterion, optimizer)

            # Validate — returns (loss, acc, kappa, auc, sens, spec, per_class)
            v_loss, v_acc, v_kappa, v_auc, v_sens, v_spec, per_class = self.validate(
                val_loader, criterion
            )

            # LR step
            scheduler.step()

            # Log scalars
            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)
            self.history["val_kappa"].append(v_kappa)
            self.history["val_auc"].append(v_auc)
            self.history["val_sens"].append(v_sens)
            self.history["val_spec"].append(v_spec)
            self.history["val_f1_macro"].append(per_class["macro_f1"])

            self.history["val_recall_per_class"].append(per_class["recall"])
            self.history["val_precision_per_class"].append(per_class["precision"])
            self.history["val_f1_per_class"].append(per_class["f1"])

            lr = optimizer.param_groups[0]["lr"]
            grade_names = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
            print(
                f"\n  Epoch {epoch + 1}/{num_epochs}\n"
                f"    Train  — loss: {t_loss:.4f}  acc: {t_acc:.4f}\n"
                f"    Val    — loss: {v_loss:.4f}  acc: {v_acc:.4f}\n"
                f"    Val κ: {v_kappa:.4f}  AUC: {v_auc:.4f}  "
                f"Sens: {v_sens:.4f}  Spec: {v_spec:.4f}  "
                f"F1-macro: {per_class['macro_f1']:.4f}  LR: {lr:.2e}\n"
                f"    Per-class F1:  "
                + "  ".join(
                    f"{name[:4]}={f1:.3f}"
                    for name, f1 in zip(grade_names, per_class["f1"])
                )
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
            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        print(f"\n{'=' * 65}")
        print(f"  Training done - Best Val kappa: {self.best_kappa:.4f}")
        print(f"{'=' * 65}\n")

        self.end_time = time.time()
        self._save_history()
        self._save_run_metrics()

    def _save_history(self):
        path = LOG_DIR / f"{self.run_tag}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  History saved to {path}")

    def _save_run_metrics(self):
        from datetime import datetime
        import time
    
        runtime_seconds = self.end_time - self.start_time if self.end_time else 0
        runtime_formatted = time.strftime("%H:%M:%S", time.gmtime(runtime_seconds))

        best_idx = self.best_epoch
        metrics = {
            "run_info": {
                "run_tag": self.run_tag,
                "model_name": self.model_name,
                "fold": self.fold,
                "timestamp": self.run_timestamp,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                "runtime_seconds": runtime_seconds,
                "runtime_formatted": runtime_formatted,
            },
            "best_metrics": {
                "epoch": self.best_epoch + 1,
                "val_kappa": self.best_kappa,
                "val_acc": self.history["val_acc"][best_idx] if best_idx < len(self.history["val_acc"]) else None,
                "val_auc": self.history["val_auc"][best_idx] if best_idx < len(self.history["val_auc"]) else None,
                "val_sens": self.history["val_sens"][best_idx] if best_idx < len(self.history["val_sens"]) else None,
                "val_spec": self.history["val_spec"][best_idx] if best_idx < len(self.history["val_spec"]) else None,
                "val_loss": self.history["val_loss"][best_idx] if best_idx < len(self.history["val_loss"]) else None,
                "train_acc": self.history["train_acc"][best_idx] if best_idx < len(self.history["train_acc"]) else None,
                "train_loss": self.history["train_loss"][best_idx] if best_idx < len(self.history["train_loss"]) else None,
                "val_f1_macro": self.history["val_f1_macro"][best_idx] if best_idx < len(self.history["val_f1_macro"]) else None,
                "val_f1_per_class": self.history["val_f1_per_class"][best_idx] if best_idx < len(self.history["val_f1_per_class"]) else None,
                "val_recall_per_class": self.history["val_recall_per_class"][best_idx] if best_idx < len(self.history["val_recall_per_class"]) else None,
                "val_precision_per_class": self.history["val_precision_per_class"][best_idx] if best_idx < len(self.history["val_precision_per_class"]) else None,
            },
            "final_metrics": {
                "epoch": len(self.history["val_kappa"]),
                "val_kappa": self.history["val_kappa"][-1] if self.history["val_kappa"] else None,
                "val_acc": self.history["val_acc"][-1] if self.history["val_acc"] else None,
                "val_auc": self.history["val_auc"][-1] if self.history["val_auc"] else None,
                "val_sens": self.history["val_sens"][-1] if self.history["val_sens"] else None,
                "val_spec": self.history["val_spec"][-1] if self.history["val_spec"] else None,
                "val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
                "train_acc": self.history["train_acc"][-1] if self.history["train_acc"] else None,
                "train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            },
            "hyperparameters": self.hyperparams,
            "training_config": {
                "epochs_requested": self.num_epochs,
                "epochs_completed": len(self.history["val_kappa"]),
                "early_stopped": len(self.history["val_kappa"]) < self.num_epochs,
                "device": str(self.device),
                "amp_enabled": self.amp_enabled,
                "grad_clip_norm": self.grad_clip_norm,
            },
            "checkpoints": {
                "best": f"{self.run_tag}_best.pth",
                "last": f"{self.run_tag}_last.pth",
            }
        }

        # Save to RESULTS_DIR
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = RESULTS_DIR / f"{self.run_tag}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to {metrics_path}")


# ================================================================================
#  Entry point


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
        help="'baseline' (ResNet-50) or 'cbam' (CBAM-ResNet50)"
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
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--use_balanced_sampler", action="store_true",
                        default=USE_BALANCED_SAMPLER)
    parser.add_argument("--classifier_hidden_dim", type=int,
                        default=CLASSIFIER_HIDDEN_DIM)
    parser.add_argument("--lr_warmup_epochs", type=int, default=LR_WARMUP_EPOCHS)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--dropout_rate", type=float, default=MC_DROPOUT_RATE)
    parser.add_argument("--early_stopping_patience", type=int,
                        default=EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--use_aug_balanced_dataset",
        action="store_true",
        default=False,
        help="Use AugmentedBalancedDataset (oversamples minority via augmentation)",
    )
    parser.add_argument("--aug_target_count_per_class", type=int, default=800)
    parser.add_argument("--aug_focal_alpha_uniform", action="store_true", default=True)
    args = parser.parse_args()

    if args.config:
        try:
            from configs.experiment_config import load_config
            args = load_config(args.config, args)
        except ImportError:
            print("Warning: Config system not available, using CLI args")

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
        use_balanced_sampler=args.use_balanced_sampler,
        use_aug_balanced_dataset=args.use_aug_balanced_dataset,
        aug_target_count_per_class=args.aug_target_count_per_class,
    )
    if args.use_aug_balanced_dataset:
        print(
            f"  AugmentedBalancedDataset active: {len(train_loader.dataset)} "
            f"training samples (target {args.aug_target_count_per_class}/class)"
        )
    if args.use_balanced_sampler:
        print("  Using WeightedRandomSampler for balanced class frequency")

    # Model Selection
    print(f"\nBuilding {args.model.upper()} model ...")
    if args.model == "baseline":
        model = create_baseline_model(
            num_classes=NUM_CLASSES,
            dropout_rate=args.dropout_rate,
            pretrained=True,
            classifier_hidden_dim=args.classifier_hidden_dim,
        )
        model_name = "baseline_resnet50"
    else:
        model = create_model(
            num_classes=NUM_CLASSES,
            dropout_rate=args.dropout_rate,
            pretrained=True,
            classifier_hidden_dim=args.classifier_hidden_dim,
        )
        model_name = "cbam_resnet50"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    train_labels = torch.tensor(train_df["diagnosis"].values)
    use_uniform_alpha = (
        (args.use_aug_balanced_dataset and args.aug_focal_alpha_uniform)
        or args.use_balanced_sampler
        or not args.use_class_weights
    )
    if use_uniform_alpha:
        alpha_weights = torch.ones(NUM_CLASSES, device=device)
        reason = (
            "AugmentedBalancedDataset + uniform flag"
            if (args.use_aug_balanced_dataset and args.aug_focal_alpha_uniform)
            else (
                "WeightedRandomSampler"
                if args.use_balanced_sampler
                else "class weights disabled"
            )
        )
        print(f"  Alpha weights: uniform [1,1,1,1,1] ({reason})")
    else:
        alpha_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
        print(f"  Class alpha weights: {alpha_weights.cpu().numpy().round(3)}")

    criterion = FocalLoss(
        gamma=FOCAL_GAMMA,
        alpha=alpha_weights,
        label_smoothing=args.label_smoothing,
    )
    if args.label_smoothing > 0.0:
        print(f"  Label smoothing: {args.label_smoothing}")

    # Optimizer + Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )

    if args.lr_warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=args.lr_warmup_epochs,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.lr_warmup_epochs],
        )
        print(f"  LR warmup: {args.lr_warmup_epochs} epochs "
              f"({args.lr * 0.01:.2e} → {args.lr:.2e})")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs,
        )

    # Trainer 
    hyperparams = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "focal_gamma": FOCAL_GAMMA,
        "image_size": IMAGE_SIZE,
        "dropout_rate": args.dropout_rate,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "optimizer": "AdamW",
        "scheduler": (
            f"SequentialLR(LinearLR({args.lr_warmup_epochs}ep) + CosineAnnealingLR)"
            if args.lr_warmup_epochs > 0
            else "CosineAnnealingLR"
        ),
        "seed": RANDOM_SEED,
        "n_folds": N_FOLDS,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "total_params": total_params,
        "label_smoothing": args.label_smoothing,
        "use_balanced_sampler": args.use_balanced_sampler,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "use_class_weights": args.use_class_weights,
        "use_aug_balanced_dataset": args.use_aug_balanced_dataset,
        "aug_target_count_per_class": args.aug_target_count_per_class,
        "aug_focal_alpha_uniform": args.aug_focal_alpha_uniform,
    }

    trainer = Trainer(
        model, device,
        fold=args.fold,
        model_name=model_name,
        grad_clip_norm=GRAD_CLIP_NORM,
        hyperparams=hyperparams,
        early_stopping_patience=args.early_stopping_patience,
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
