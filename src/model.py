"""
Uncertainty-Aware Attention CNN — Model Architecture.

Architecture
------------
ResNet-50  (ImageNet-pretrained backbone)
  └─ CBAM  injected after layer1, layer2, layer3, layer4
  └─ Global Average Pool
  └─ MCDropout(p=0.5)   ← always stochastic, even in eval()
  └─ Linear(2048  →  5)

Key design decisions
--------------------
1. **CBAM placement**: Attached *after* each stage (not inside residual
   blocks) so we don't break pretrained weight compatibility.
   Each CBAM independently re-calibrates the feature map through
   channel attention (what) and spatial attention (where).

2. **MCDropout**: A custom ``nn.Module`` that calls
   ``F.dropout(x, p, training=True)`` — the ``training=True`` flag is
   *hard-coded*, so dropout stays active even when the model is in
   ``model.eval()`` mode.  This lets BatchNorm use running stats
   (stable predictions) while still sampling from the approximate
   Bayesian posterior (uncertainty quantification).

3. **Single-layer head**: The final classifier is intentionally minimal
   (Dropout → Linear) to avoid adding learnable capacity that could
   overfit on the small APTOS dataset (~3.6 k images).

Reference:
  CBAM — Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

from config import (
    NUM_CLASSES,
    MC_DROPOUT_RATE,
    CBAM_REDUCTION_RATIO,
)


# =========================================================================
#  CBAM components
# =========================================================================

class ChannelAttention(nn.Module):
    """Channel attention sub-module.

    Applies average-pool  AND  max-pool over the spatial dims, feeds each
    through a shared two-layer MLP  (C → C/r → C), then sums and sigmoids.

    Args:
        in_channels: Number of input feature channels.
        reduction: Bottleneck reduction ratio *r*.
    """

    def __init__(self, in_channels: int, reduction: int = CBAM_REDUCTION_RATIO):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        # Shared MLP implemented as two 1×1 convolutions (equivalent to Linear)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Attention-weighted x, same shape.
        """
        b, c, _, _ = x.size()

        # Global pools → (B, C)
        avg_pool = x.mean(dim=[2, 3])               # (B, C)
        max_pool = x.amax(dim=[2, 3])                # (B, C)

        # Shared MLP
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        # Combine + gate
        scale = torch.sigmoid(avg_out + max_out)     # (B, C)
        return x * scale.unsqueeze(-1).unsqueeze(-1)  # broadcast to (B, C, H, W)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module.

    Computes avg and max across the channel dim, concatenates them into a
    2-channel map, then applies a 7×7 conv → sigmoid to produce a spatial
    gate of shape ``(B, 1, H, W)``.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = x.mean(dim=1, keepdim=True)       # (B, 1, H, W)
        max_pool = x.amax(dim=1, keepdim=True)        # (B, 1, H, W)

        combined = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        scale = torch.sigmoid(self.conv(combined))          # (B, 1, H, W)
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Sequential composition: ChannelAttention → SpatialAttention.
    """

    def __init__(self, in_channels: int, reduction: int = CBAM_REDUCTION_RATIO):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# =========================================================================
#  MC Dropout (always-on)
# =========================================================================

class MCDropout(nn.Module):
    """Dropout layer that stays active during ``model.eval()``.

    Standard ``nn.Dropout`` checks ``self.training`` and becomes a no-op
    in eval mode.  Here we hard-code ``training=True`` in the functional
    call so the mask is always sampled — this is the key mechanism for
    Monte-Carlo Dropout Bayesian inference.
    """

    def __init__(self, p: float = MC_DROPOUT_RATE):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # training=True is intentional — dropout is ALWAYS active
        return F.dropout(x, self.p, training=True)


# =========================================================================
#  Full model
# =========================================================================

class CBAMResNet50(nn.Module):
    """ResNet-50 + CBAM attention at every stage + MC Dropout Bayesian head.

    Architecture::

        Input (B, 3, 512, 512)
          │
          ├─ ResNet stem  (conv1 + bn1 + relu + maxpool)
          ├─ layer1  →  CBAM(256)
          ├─ layer2  →  CBAM(512)
          ├─ layer3  →  CBAM(1024)
          ├─ layer4  →  CBAM(2048)
          │
          ├─ AdaptiveAvgPool2d → (B, 2048)
          ├─ MCDropout(p=0.5)          ← always-on stochastic
          └─ Linear(2048, 5)

    Args:
        num_classes: Output logits dimension (5 for DR grading).
        dropout_rate: MC Dropout probability.
        pretrained: Use ImageNet-pretrained weights.
        reduction: CBAM channel-attention reduction ratio.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = MC_DROPOUT_RATE,
        pretrained: bool = True,
        reduction: int = CBAM_REDUCTION_RATIO,
    ):
        super().__init__()

        # --- Backbone ---
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Extract stages (we do NOT keep backbone.fc / backbone.avgpool)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1     # out: 256  channels
        self.layer2 = backbone.layer2     # out: 512  channels
        self.layer3 = backbone.layer3     # out: 1024 channels
        self.layer4 = backbone.layer4     # out: 2048 channels

        # --- CBAM after each stage ---
        self.cbam1 = CBAM(256,  reduction)
        self.cbam2 = CBAM(512,  reduction)
        self.cbam3 = CBAM(1024, reduction)
        self.cbam4 = CBAM(2048, reduction)

        # --- Bayesian classification head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mc_dropout = MCDropout(dropout_rate)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  Returns raw logits ``(B, num_classes)``."""
        x = self.stem(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)           # (B, 2048)
        x = self.mc_dropout(x)            # always stochastic
        x = self.fc(x)                    # (B, num_classes)
        return x


# =========================================================================
#  Factory + smoke test
# =========================================================================

def create_model(
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = MC_DROPOUT_RATE,
    pretrained: bool = True,
) -> CBAMResNet50:
    """Factory function used by train.py and evaluate.py."""
    return CBAMResNet50(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    # --- Shape test ---
    model = CBAMResNet50(num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Output shape: {out.shape}")    # (2, 5)

    # --- MC Dropout stays active in eval mode ---
    model.eval()
    o1, o2 = model(x), model(x)
    assert not torch.equal(o1, o2), "MC Dropout must produce different outputs!"
    print("MC Dropout active in eval() ✓")

    # --- Parameter count ---
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print("All model tests passed ✓")
