"""Unit tests for model components."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model import MCDropout, CBAMResNet50, BaselineResNet50, create_model, create_baseline_model


def test_mcdropout_deterministic_mode():
    """Verify deterministic mode produces identical outputs."""
    model = create_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        o1 = model(x)
        o2 = model(x)
    assert not torch.allclose(o1, o2), "MC mode should produce different outputs"

    with torch.no_grad():
        with model.deterministic_mode():
            d1 = model(x)
            d2 = model(x)
    assert torch.allclose(d1, d2), "Deterministic mode should produce identical outputs"

    with torch.no_grad():
        o3 = model(x)
        o4 = model(x)
    assert not torch.allclose(o3, o4), "MC mode should be restored after context exit"

    print("MCDropout deterministic mode test passed!")


def test_baseline_model():
    """Verify baseline model architecture."""
    model = create_baseline_model(pretrained=False)
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"

    param_count = sum(p.numel() for p in model.parameters())
    assert 23_000_000 < param_count < 24_000_000, f"Unexpected param count: {param_count:,}"

    print(f"Baseline model test passed! Params: {param_count:,}")


def test_baseline_deterministic_mode():
    """Verify baseline deterministic mode works."""
    model = create_baseline_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        with model.deterministic_mode():
            d1 = model(x)
            d2 = model(x)
    assert torch.allclose(d1, d2), "Baseline deterministic mode should work"

    print("Baseline deterministic mode test passed!")


def test_cbam_has_more_params():
    """Verify CBAM model has more parameters than baseline."""
    cbam = create_model(pretrained=False)
    baseline = create_baseline_model(pretrained=False)

    cbam_params = sum(p.numel() for p in cbam.parameters())
    baseline_params = sum(p.numel() for p in baseline.parameters())

    assert cbam_params > baseline_params, "CBAM should have more params"
    print(f"CBAM adds {cbam_params - baseline_params:,} parameters")


if __name__ == "__main__":
    test_mcdropout_deterministic_mode()
    test_baseline_model()
    test_baseline_deterministic_mode()
    test_cbam_has_more_params()
    print("\nAll model tests passed!")
