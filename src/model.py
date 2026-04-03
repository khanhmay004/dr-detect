from contextlib import contextmanager
from typing import Iterator

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



#  CBAM components

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = CBAM_REDUCTION_RATIO):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    # KERNEL size = 7 ->> 3?
    def __init__(self, kernel_size: int = 3):
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
    def __init__(self, in_channels: int, reduction: int = CBAM_REDUCTION_RATIO):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# =========================================================================
#  MC Dropout

class MCDropout(nn.Module):
    def __init__(self, p: float = MC_DROPOUT_RATE):
        super().__init__()
        self.p = p
        self.mc_active = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mc_active:
            return F.dropout(x, self.p, training=True)
        return x


# =========================================================================
#  Full model

class CBAMResNet50(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = MC_DROPOUT_RATE,
        pretrained: bool = True,
        reduction: int = CBAM_REDUCTION_RATIO,
        classifier_hidden_dim: int = 0,
    ):
        super().__init__()

        # --- Backbone ---
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1     # out: 256  channels
        self.layer2 = backbone.layer2     # out: 512  channels
        self.layer3 = backbone.layer3     # out: 1024 channels
        self.layer4 = backbone.layer4     # out: 2048 channels

        # CBAM after each stage
        self.cbam1 = CBAM(256,  reduction)
        self.cbam2 = CBAM(512,  reduction)
        self.cbam3 = CBAM(1024, reduction)
        self.cbam4 = CBAM(2048, reduction)

        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        if self._use_classifier_seq:
            x = self.classifier(x)         # (B, num_classes)
        else:
            x = self.mc_dropout(x)
            x = self.fc(x)                 # (B, num_classes)
        return x

    @contextmanager
    def deterministic_mode(self) -> Iterator[None]:
        # Temporarily disable MC Dropout for deterministic val
        mc_modules = [m for m in self.modules() if isinstance(m, MCDropout)]
        for m in mc_modules:
            m.mc_active = False
        try:
            yield
        finally:
            for m in mc_modules:
                m.mc_active = True



def create_model(
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = MC_DROPOUT_RATE,
    pretrained: bool = True,
    classifier_hidden_dim: int = 0,
) -> CBAMResNet50:
    """Factory for CBAM-ResNet50.

    Args:
        num_classes: Number of output classes.
        dropout_rate: MC Dropout probability.
        pretrained: Use ImageNet pretrained backbone.
        classifier_hidden_dim: Hidden dim for deeper head (0 = original head).

    Returns:
        CBAMResNet50 instance.
    """
    return CBAMResNet50(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        classifier_hidden_dim=classifier_hidden_dim,
    )


#  Baseline ResNet-50 

class BaselineResNet50(nn.Module):

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = MC_DROPOUT_RATE,
        pretrained: bool = True,
        classifier_hidden_dim: int = 0,
    ):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
            self.mc_dropout = MCDropout(dropout_rate)
            self.fc = nn.Linear(2048, num_classes)
            self._use_classifier_seq = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)           # (B, 2048)
        if self._use_classifier_seq:
            x = self.classifier(x)         # (B, num_classes)
        else:
            x = self.mc_dropout(x)
            x = self.fc(x)                 # (B, num_classes)
        return x

    @contextmanager
    def deterministic_mode(self) -> Iterator[None]:
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
    classifier_hidden_dim: int = 0,
) -> BaselineResNet50:
    """Factory for BaselineResNet50.

    Args:
        num_classes: Number of output classes.
        dropout_rate: MC Dropout probability.
        pretrained: Use ImageNet pretrained backbone.
        classifier_hidden_dim: Hidden dim for deeper head (0 = original head).

    Returns:
        BaselineResNet50 instance.
    """
    return BaselineResNet50(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        classifier_hidden_dim=classifier_hidden_dim,
    )







if __name__ == "__main__":
    x = torch.randn(2, 3, 512, 512)

    # --- CBAM Model (original head, hidden_dim=0) ---
    model = CBAMResNet50(num_classes=5, pretrained=False)
    out = model(x)
    print(f"CBAM (hidden_dim=0) output shape: {out.shape}")

    model.eval()
    o1, o2 = model(x), model(x)
    assert not torch.equal(o1, o2), "MC Dropout must produce different outputs!"
    print("MC Dropout active in eval() ✓")

    with model.deterministic_mode():
        d1 = model(x)
        d2 = model(x)
    assert torch.equal(d1, d2), "Deterministic mode should produce identical outputs!"
    print("Deterministic mode works ✓")

    o3, o4 = model(x), model(x)
    assert not torch.equal(o3, o4), "MC mode should be restored after context exit!"
    print("MC mode restored after context exit ✓")

    cbam_params = sum(p.numel() for p in model.parameters())
    print(f"CBAM params (hidden_dim=0): {cbam_params:,}")

    # --- CBAM Model (deeper head, hidden_dim=512) ---
    model_deep = CBAMResNet50(num_classes=5, pretrained=False, classifier_hidden_dim=512)
    out_deep = model_deep(x)
    print(f"\nCBAM (hidden_dim=512) output shape: {out_deep.shape}")

    model_deep.eval()
    with model_deep.deterministic_mode():
        dd1 = model_deep(x)
        dd2 = model_deep(x)
    assert torch.equal(dd1, dd2), "Deeper head deterministic mode should work!"
    print("Deeper head deterministic mode works ✓")

    deep_params = sum(p.numel() for p in model_deep.parameters())
    print(f"CBAM params (hidden_dim=512): {deep_params:,}")
    print(f"Deeper head adds {deep_params - cbam_params:,} parameters")

    # --- Baseline Model (original head) ---
    baseline = BaselineResNet50(num_classes=5, pretrained=False)
    baseline.eval()
    out_b = baseline(x)
    print(f"\nBaseline (hidden_dim=0) output shape: {out_b.shape}")

    with baseline.deterministic_mode():
        bd1 = baseline(x)
        bd2 = baseline(x)
    assert torch.equal(bd1, bd2), "Baseline deterministic mode should work!"
    print("Baseline deterministic mode works ✓")

    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Baseline params (hidden_dim=0): {baseline_params:,}")

    # --- Baseline Model (deeper head) ---
    baseline_deep = BaselineResNet50(num_classes=5, pretrained=False, classifier_hidden_dim=512)
    baseline_deep.eval()
    out_bd = baseline_deep(x)
    print(f"\nBaseline (hidden_dim=512) output shape: {out_bd.shape}")

    with baseline_deep.deterministic_mode():
        bdd1 = baseline_deep(x)
        bdd2 = baseline_deep(x)
    assert torch.equal(bdd1, bdd2), "Baseline deeper head deterministic mode should work!"
    print("Baseline deeper head deterministic mode works ✓")

    baseline_deep_params = sum(p.numel() for p in baseline_deep.parameters())
    print(f"Baseline params (hidden_dim=512): {baseline_deep_params:,}")

    print(f"\nCBAM adds {cbam_params - baseline_params:,} parameters (original head)")
    print("All model tests passed ✓")
