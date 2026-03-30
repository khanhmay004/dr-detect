"""Experiment configuration system for DR detection pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_WORKERS,
    IMAGE_SIZE, MC_DROPOUT_RATE, FOCAL_GAMMA, GRAD_CLIP_NORM,
    RANDOM_SEED, N_FOLDS, EARLY_STOPPING_PATIENCE,
)


@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment."""

    name: str = "default"
    description: str = ""

    model_type: str = "cbam"
    pretrained: bool = True
    dropout_rate: float = MC_DROPOUT_RATE

    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    batch_size: int = BATCH_SIZE
    grad_clip_norm: float = GRAD_CLIP_NORM
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE

    focal_gamma: float = FOCAL_GAMMA
    use_class_weights: bool = True

    image_size: int = IMAGE_SIZE
    num_workers: int = NUM_WORKERS
    fold: int = 0
    n_folds: int = N_FOLDS

    device: str = "auto"
    use_amp: bool = True

    seed: int = RANDOM_SEED

    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        import yaml
        with open(yaml_path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def cpu_smoke_test_config() -> ExperimentConfig:
    """Minimal config for CPU testing."""
    return ExperimentConfig(
        name="cpu_smoke_test",
        description="Quick CPU test to verify code runs correctly",
        model_type="baseline",
        epochs=2,
        batch_size=2,
        num_workers=0,
        device="cpu",
        use_amp=False,
        image_size=256,
    )


def gpu_baseline_config() -> ExperimentConfig:
    """Full training config for baseline ResNet-50."""
    return ExperimentConfig(
        name="baseline_resnet50_full",
        description="Baseline ResNet-50 training on GPU",
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
        description="CBAM-ResNet50 training on GPU",
        model_type="cbam",
        epochs=20,
        batch_size=16,
        num_workers=4,
        device="cuda",
        use_amp=True,
    )


def load_config(config_path: str, cli_args: Any) -> Any:
    """Load config from file and merge with CLI arguments."""
    config = ExperimentConfig.from_yaml(config_path)

    cli_args.epochs = config.epochs
    cli_args.lr = config.learning_rate
    cli_args.batch_size = config.batch_size
    cli_args.fold = config.fold
    cli_args.model = config.model_type

    return cli_args
