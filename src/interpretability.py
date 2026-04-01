"""
Grad-CAM interpretability module for DR detection models.

Provides lesion-focused heatmaps from trained checkpoints with side-by-side
visualization for clinical trust assessment.

Key components:
    - GradCAM: Hook-based gradient-weighted class activation mapping
    - Overlay utilities: Heatmap blending with original images
    - Panel plotting: Multi-panel figure export for thesis

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017
"""

from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DR_GRADES,
    DR_COLORS,
    FIGURES_DIR,
)


class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Generates saliency maps highlighting regions most relevant to a
    specific class prediction. Uses forward/backward hooks to capture
    intermediate activations and gradients.

    Args:
        model: Trained CNN model (BaselineResNet50 or CBAMResNet50).
        target_module: The convolutional layer to extract CAM from.

    Example:
        >>> model = create_model()
        >>> gradcam = GradCAM(model, model.layer4)
        >>> cam = gradcam.generate(input_tensor, class_index=2)
        >>> gradcam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target module."""

        def forward_hook(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            self.activations = output.detach()

        def backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            self.gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_module.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_module.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_index: int | None = None,
        upsample_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Generate Grad-CAM heatmap for the input.

        Args:
            input_tensor: Input image tensor, shape (1, 3, H, W).
            class_index: Target class for CAM. If None, uses predicted class.
            upsample_size: Output size (H, W). If None, uses input size.

        Returns:
            CAM heatmap tensor, shape (H, W), values in [0, 1].
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError("Input must have shape (1, 3, H, W)")

        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        score = logits[0, class_index]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Ensure target_module is part of the forward pass."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        if upsample_size is None:
            upsample_size = (input_tensor.size(2), input_tensor.size(3))

        cam = F.interpolate(
            cam,
            size=upsample_size,
            mode="bilinear",
            align_corners=False,
        )

        return cam.squeeze().cpu()


def get_target_module(
    model: nn.Module,
    layer_name: str,
) -> nn.Module:
    """Retrieve target module from model by name.

    Args:
        model: The model instance.
        layer_name: One of 'layer3', 'layer4', 'cbam3', 'cbam4'.

    Returns:
        The target nn.Module for Grad-CAM hooks.

    Raises:
        ValueError: If layer_name is invalid or not found.
    """
    valid_layers = ["layer3", "layer4", "cbam3", "cbam4"]
    if layer_name not in valid_layers:
        raise ValueError(f"layer_name must be one of {valid_layers}")

    if not hasattr(model, layer_name):
        if layer_name.startswith("cbam"):
            raise ValueError(
                f"Model does not have '{layer_name}'. "
                "CBAM layers are only available in CBAMResNet50."
            )
        raise ValueError(f"Model does not have attribute '{layer_name}'")

    return getattr(model, layer_name)


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> np.ndarray:
    """Convert normalized tensor back to RGB uint8 image.

    Args:
        tensor: Normalized image tensor, shape (3, H, W) or (1, 3, H, W).
        mean: Normalization mean (ImageNet default).
        std: Normalization std (ImageNet default).

    Returns:
        RGB image as uint8 ndarray, shape (H, W, 3).
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = tensor.cpu().clone()

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std_t + mean_t

    tensor = torch.clamp(tensor, 0.0, 1.0)
    image = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return image


def apply_colormap(
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Apply colormap to grayscale CAM.

    Args:
        cam: Grayscale CAM, shape (H, W), values in [0, 1].
        colormap: OpenCV colormap constant.

    Returns:
        Colored heatmap as RGB uint8 ndarray, shape (H, W, 3).
    """
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend CAM heatmap with original image.

    Args:
        image: RGB image, shape (H, W, 3), uint8.
        cam: Grayscale CAM, shape (H, W), values in [0, 1].
        alpha: Heatmap opacity (0 = image only, 1 = heatmap only).

    Returns:
        Blended RGB image as uint8 ndarray, shape (H, W, 3).
    """
    if cam.shape[:2] != image.shape[:2]:
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

    heatmap = apply_colormap(cam)
    blended = (alpha * heatmap + (1.0 - alpha) * image).astype(np.uint8)

    return blended


def create_gradcam_panel(
    image: np.ndarray,
    cam: np.ndarray,
    true_grade: int,
    pred_grade: int,
    confidence: float,
    entropy: float,
    image_id: str,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Create multi-panel Grad-CAM visualization.

    Generates a figure with three panels:
        1. Original image
        2. Grad-CAM heatmap
        3. Overlay with metadata

    Args:
        image: RGB image, shape (H, W, 3), uint8.
        cam: Grayscale CAM, shape (H, W), values in [0, 1].
        true_grade: Ground truth DR grade (0-4 or -1 if unknown).
        pred_grade: Predicted DR grade (0-4).
        confidence: Prediction confidence (max probability).
        entropy: Predictive entropy from MC Dropout.
        image_id: Image identifier for title.
        save_path: If provided, saves figure to this path.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    heatmap = apply_colormap(cam)
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    overlay = overlay_cam_on_image(image, cam, alpha=0.45)
    axes[2].imshow(overlay)

    true_label = DR_GRADES.get(true_grade, "Unknown")
    pred_label = DR_GRADES.get(pred_grade, "Unknown")
    pred_color = DR_COLORS.get(pred_grade, "#333333")

    correct = true_grade == pred_grade
    status = "Correct" if correct else "Incorrect"
    status_color = "#2ecc71" if correct else "#e74c3c"

    title_text = (
        f"Overlay | {status}\n"
        f"True: {true_label} | Pred: {pred_label}\n"
        f"Conf: {confidence:.2f} | Entropy: {entropy:.3f}"
    )
    axes[2].set_title(title_text, fontsize=10, color=status_color)
    axes[2].axis("off")

    fig.suptitle(
        f"Grad-CAM Analysis: {image_id}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_comparison_panel(
    image: np.ndarray,
    cam_baseline: np.ndarray,
    cam_cbam: np.ndarray,
    true_grade: int,
    pred_baseline: int,
    pred_cbam: int,
    conf_baseline: float,
    conf_cbam: float,
    image_id: str,
    save_path: Path | None = None,
    figsize: tuple[float, float] = (16, 4),
) -> plt.Figure:
    """Create side-by-side baseline vs CBAM comparison panel.

    Args:
        image: RGB image, shape (H, W, 3), uint8.
        cam_baseline: Baseline model CAM, shape (H, W).
        cam_cbam: CBAM model CAM, shape (H, W).
        true_grade: Ground truth DR grade.
        pred_baseline: Baseline prediction.
        pred_cbam: CBAM prediction.
        conf_baseline: Baseline confidence.
        conf_cbam: CBAM confidence.
        image_id: Image identifier.
        save_path: If provided, saves figure to this path.
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(image)
    true_label = DR_GRADES.get(true_grade, "Unknown")
    axes[0].set_title(f"Original\nTrue: {true_label}", fontsize=10)
    axes[0].axis("off")

    overlay_baseline = overlay_cam_on_image(image, cam_baseline, alpha=0.45)
    axes[1].imshow(overlay_baseline)
    pred_label_b = DR_GRADES.get(pred_baseline, "Unknown")
    correct_b = true_grade == pred_baseline
    color_b = "#2ecc71" if correct_b else "#e74c3c"
    axes[1].set_title(
        f"Baseline (layer4)\nPred: {pred_label_b} ({conf_baseline:.2f})",
        fontsize=10,
        color=color_b,
    )
    axes[1].axis("off")

    overlay_cbam = overlay_cam_on_image(image, cam_cbam, alpha=0.45)
    axes[2].imshow(overlay_cbam)
    pred_label_c = DR_GRADES.get(pred_cbam, "Unknown")
    correct_c = true_grade == pred_cbam
    color_c = "#2ecc71" if correct_c else "#e74c3c"
    axes[2].set_title(
        f"CBAM (layer4)\nPred: {pred_label_c} ({conf_cbam:.2f})",
        fontsize=10,
        color=color_c,
    )
    axes[2].axis("off")

    diff = np.abs(cam_cbam - cam_baseline)
    diff_colored = apply_colormap(diff)
    axes[3].imshow(diff_colored)
    axes[3].set_title("Attention Difference\n|CBAM - Baseline|", fontsize=10)
    axes[3].axis("off")

    fig.suptitle(
        f"Model Comparison: {image_id}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def select_case_ids(
    uncertainty_df,
    n_per_bucket: int = 5,
) -> dict[str, list[str]]:
    """Select representative cases for Grad-CAM analysis.

    Creates four analysis buckets:
        1. Correct + low entropy (confident correct predictions)
        2. Correct + high entropy (uncertain but correct)
        3. Incorrect + high entropy (uncertain mistakes)
        4. Referable false negatives (dangerous misses)

    Args:
        uncertainty_df: DataFrame with columns:
            image_id, true_grade, predicted_grade, entropy, confidence
        n_per_bucket: Number of samples per bucket.

    Returns:
        Dict mapping bucket names to lists of image IDs.
    """
    df = uncertainty_df.copy()

    if "true_grade" in df.columns and "predicted_grade" in df.columns:
        df["correct"] = df["true_grade"] == df["predicted_grade"]
    else:
        raise ValueError(
            "DataFrame must have 'true_grade' and 'predicted_grade' columns"
        )

    if "entropy" not in df.columns:
        raise ValueError("DataFrame must have 'entropy' column")

    entropy_q75 = df["entropy"].quantile(0.75)
    entropy_q25 = df["entropy"].quantile(0.25)

    buckets = {
        "correct_low_entropy": df[
            df["correct"] & (df["entropy"] <= entropy_q25)
        ].nsmallest(n_per_bucket, "entropy"),
        "correct_high_entropy": df[
            df["correct"] & (df["entropy"] >= entropy_q75)
        ].nlargest(n_per_bucket, "entropy"),
        "incorrect_high_entropy": df[
            (~df["correct"]) & (df["entropy"] >= entropy_q75)
        ].nlargest(n_per_bucket, "entropy"),
        "referable_false_negative": df[
            (df["true_grade"] >= 2) & (df["predicted_grade"] < 2)
        ].nlargest(n_per_bucket, "entropy"),
    }

    result = {}
    for name, subset in buckets.items():
        ids = subset["image_id"].astype(str).tolist()
        result[name] = ids[:n_per_bucket]

    return result


def select_by_criteria(
    uncertainty_df,
    selection: str,
    n_images: int = 20,
    seed: int = 42,
) -> list[str]:
    """Select image IDs based on specified criteria.

    Args:
        uncertainty_df: DataFrame with uncertainty metrics.
        selection: One of 'high_entropy', 'low_entropy', 'random',
            'misclassified', 'false_negative'.
        n_images: Number of images to select.
        seed: Random seed for reproducibility.

    Returns:
        List of selected image IDs.
    """
    df = uncertainty_df.copy()

    if "true_grade" in df.columns and "predicted_grade" in df.columns:
        df["correct"] = df["true_grade"] == df["predicted_grade"]
    elif selection in ["misclassified", "false_negative"]:
        raise ValueError(
            f"Selection '{selection}' requires 'true_grade' and "
            "'predicted_grade' columns"
        )

    if selection == "high_entropy":
        subset = df.nlargest(n_images, "entropy")
    elif selection == "low_entropy":
        subset = df.nsmallest(n_images, "entropy")
    elif selection == "random":
        subset = df.sample(n=min(n_images, len(df)), random_state=seed)
    elif selection == "misclassified":
        subset = df[~df["correct"]].nlargest(n_images, "entropy")
    elif selection == "false_negative":
        fn_mask = (df["true_grade"] >= 2) & (df["predicted_grade"] < 2)
        subset = df[fn_mask].nlargest(n_images, "entropy")
    else:
        raise ValueError(
            f"Unknown selection: '{selection}'. "
            "Must be one of: high_entropy, low_entropy, random, "
            "misclassified, false_negative"
        )

    return subset["image_id"].astype(str).tolist()


if __name__ == "__main__":
    print("Interpretability module loaded successfully.")
    print(f"Output directory: {FIGURES_DIR / 'gradcam'}")
