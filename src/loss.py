import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import FOCAL_GAMMA


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
        """
        tính standard CE cho mỗi sample reduction=none -> tính p_t = exp(-CE) -> tính focal weight (1-p_t)^gamma -> nhân focal weight vào CE -> nhân alpha nếu có -> cuối cùng mới reduction
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # p_t: probability assigned to the true class 
        # p_t = exp(-CE)
        p_t = torch.exp(-ce_loss)

        # cong thuc focal loss: (1-p_t)^gamma * CE
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        # per class weighting (alpha)
        if self.alpha is not None:
            # Gather alpha for each sample's true class
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        # reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def compute_class_weights(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1)) #class with fewer samples get higher weight, avoid chia by zero
    return weights


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(8, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])

    fl = FocalLoss(gamma=2.0)
    loss = fl(logits, targets)
    loss.backward()

    print(f"Focal Loss: {loss.item():.4f}")
    print(f"Grad norm:  {logits.grad.norm().item():.4f}")
    print("Pass check.")
