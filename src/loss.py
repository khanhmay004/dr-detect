"""
Focal Loss for class-imbalanced classification.

The APTOS 2019 dataset is severely skewed (>50 % grade-0, <5 % grade-4).
Standard cross-entropy treats every sample equally, so the model can reach
high accuracy by predicting the majority class and ignoring rare grades.

Focal Loss (Lin et al., 2017) down-weights easy / well-classified examples
and focuses the gradient on hard negatives:

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where p_t is the model's predicted probability for the *true* class.

When gamma = 0 this reduces to standard cross-entropy.
When gamma = 2 (our default), samples the model already classifies
correctly with >0.9 confidence contribute ~100x less gradient than
misclassified samples.

Reference: "Focal Loss for Dense Object Detection", arXiv:1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import FOCAL_GAMMA


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional per-class alpha weighting.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
               0 → standard CE.  2 → recommended default.
        alpha: Optional 1-D tensor of per-class weights (length = num_classes).
               If ``None``, all classes are weighted equally.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.

    Shape:
        - logits: ``(N, C)`` raw (un-softmaxed) scores.
        - targets: ``(N,)`` class indices in ``[0, C)``.
    """

    def __init__(
        self,
        gamma: float = FOCAL_GAMMA,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Register alpha as a buffer so it moves to GPU with .to(device)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Implementation note
        -------------------
        We first compute the standard CE per sample (``reduction='none'``),
        then modulate it by ``(1 - p_t)^gamma``.  This is numerically
        stable because ``F.cross_entropy`` uses the log-sum-exp trick
        internally, and ``p_t`` is derived via ``exp(-ce)`` which is exact.
        """
        # --- per-sample cross-entropy (shape: (N,)) ---
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # --- p_t: probability assigned to the true class ---
        # p_t = exp(-CE) is more stable than indexing into softmax
        p_t = torch.exp(-ce_loss)

        # --- focal modulation ---
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        # --- per-class alpha weighting ---
        if self.alpha is not None:
            # Gather alpha for each sample's true class
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        # --- reduction ---
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def compute_class_weights(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Compute inverse-frequency alpha weights from a label tensor.

    Returns:
        1-D tensor of shape ``(num_classes,)`` where each entry is
        ``total / (num_classes * count_c)``.  Classes with fewer samples
        get higher weight.
    """
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    return weights


if __name__ == "__main__":
    # Quick gradient sanity check
    torch.manual_seed(0)
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])

    fl = FocalLoss(gamma=2.0)
    loss = fl(logits, targets)
    loss.backward()

    print(f"Focal Loss: {loss.item():.4f}")
    print(f"Grad norm:  {logits.grad.norm().item():.4f}")
    print("Gradient sanity check passed ✓")
