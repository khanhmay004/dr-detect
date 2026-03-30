"""Test validation determinism with MCDropout deterministic mode."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model import create_baseline_model


def test_validation_determinism():
    """Run forward pass 3 times with deterministic mode, verify identical results."""
    model = create_baseline_model(pretrained=False)
    model.eval()

    x = torch.randn(4, 3, 64, 64)

    outputs = []
    for i in range(3):
        with torch.no_grad():
            with model.deterministic_mode():
                out = model(x)
                outputs.append(out.clone())
        print(f"  Run {i+1}: output sum = {out.sum().item():.6f}")

    for i in range(1, 3):
        assert torch.equal(outputs[0], outputs[i]), \
            f"Run {i+1} differs from Run 1!"

    print("Validation determinism test passed!")


if __name__ == "__main__":
    test_validation_determinism()
