# Implementation Plan: Grad-CAM / Attention Map Interpretability for Clinical Trust

> **Document Version**: 1.1
> **Created**: 2026-03-31
> **Updated**: 2026-04-01
> **Status**: ✅ IMPLEMENTED
> **Scope**: Add Grad-CAM-based visual interpretability for baseline and CBAM models, integrated with current Phase 2/3 pipeline.

---

## Executive Summary

This plan adds an interpretability module that produces lesion-focused heatmaps from trained checkpoints, with side-by-side visualization for:

1. Raw fundus image
2. Grad-CAM overlay
3. Predicted class and confidence
4. Uncertainty statistics (entropy/confidence from existing MC pipeline)

Primary goal: improve **clinical trustworthiness** by showing where the model attends when making DR grading decisions, especially for high-stakes classes (grades 2–4) and uncertain cases.

---

## 1) Why this matters for this project

Based on `docs/research-project.md`:

- Current architecture includes CBAM attention but lacks explicit visual interpretability outputs.
- Phase 2 shows baseline currently outperforming CBAM on fold-0 and Messidor-2.
- Phase 3 focuses calibration/referral; interpretability is explicitly listed in future work and can strengthen thesis discussion regardless of CBAM win/loss.

Interpretability contribution:

- Supports clinical plausibility checks (lesion-relevant focus vs. spurious focus).
- Provides qualitative evidence for/against CBAM benefit.
- Improves thesis defensibility when quantitative gains are modest.

---

## 2) Design decisions

## 2.1 Target layers for Grad-CAM

Use model stage outputs that align with architecture and lesion scale:

- `layer4` (coarse semantic localization; default standard)
- Optional: `layer3` (finer localization than layer4)

For CBAM model, support both:

- post-backbone stage (`layer4`)
- post-attention stage (`cbam4`) for attention-aware contrast

Recommended default for first implementation:

- Grad-CAM on `layer4` for baseline and CBAM to keep apples-to-apples comparison.

## 2.2 Deterministic mode for reproducibility

Because `MCDropout` is active in eval by default, Grad-CAM should run with deterministic behavior to avoid stochastic maps:

- Use `with model.deterministic_mode():` for both baseline and CBAM.

## 2.3 Input source

Use existing Messidor and APTOS dataset loaders and preprocessing, then save overlays in:

- `outputs/figures/gradcam/`

---

## 3) Proposed code structure

Keep flat structure (consistent with project style):

```text
src/
├── interpretability.py       # new: Grad-CAM core + plotting utilities
└── run_gradcam.py            # new: CLI entry point for batch generation
```

No deep subfolders; reuse existing config/model/dataset modules.

---

## 4) Implementation tasks (phased)

## Phase I — Core Grad-CAM engine

- [x] Add `src/interpretability.py`
- [x] Implement forward/backward hook manager
- [x] Implement Grad-CAM heatmap computation
- [x] Implement overlay rendering helper

### Reference implementation snippet

```python
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations = output.detach()  # [B, C, H, W]

        def backward_hook(_: torch.nn.Module, grad_input, grad_output) -> None:
            self.gradients = grad_output[0].detach()  # [B, C, H, W]

        self._hooks.append(self.target_module.register_forward_hook(forward_hook))
        self._hooks.append(self.target_module.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def generate(self, input_tensor: torch.Tensor, class_index: int | None = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)  # [B, 5]
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        score = logits[:, class_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Global-average gradients over spatial dims to get channel weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam
```

---

## Phase II — Visualization and export

- [x] Add helper to convert normalized tensor back to RGB view
- [x] Add colormap overlay utility
- [x] Add multi-panel figure export:
  - original image
  - heatmap
  - overlay
  - metadata text (true/pred/conf/entropy)

### Overlay snippet

```python
def overlay_cam_on_rgb(
    rgb_image: np.ndarray,   # [H, W, 3], uint8
    cam_map: np.ndarray,     # [H, W], float32 in [0, 1]
    alpha: float = 0.45,
) -> np.ndarray:
    heatmap = plt.get_cmap("jet")(cam_map)[..., :3]  # [H, W, 3]
    heatmap_uint8 = (heatmap * 255.0).astype(np.uint8)
    blended = (alpha * heatmap_uint8 + (1.0 - alpha) * rgb_image).astype(np.uint8)
    return blended
```

---

## Phase III — CLI integration (`run_gradcam.py`)

- [x] Build CLI for model/checkpoint/dataset selection
- [x] Support baseline and CBAM with shared interface
- [x] Support deterministic Grad-CAM generation
- [x] Support selecting:
  - fixed image IDs
  - top-N high entropy samples
  - random stratified samples by grade

### CLI skeleton

```python
parser = argparse.ArgumentParser(description="Grad-CAM visualization for DR models")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--model", type=str, choices=["baseline", "cbam"], required=True)
parser.add_argument("--dataset", type=str, choices=["aptos_val", "messidor2"], default="messidor2")
parser.add_argument("--target_layer", type=str, default="layer4",
                    choices=["layer3", "layer4", "cbam4"])
parser.add_argument("--num_images", type=int, default=20)
parser.add_argument("--selection", type=str, default="high_entropy",
                    choices=["high_entropy", "low_entropy", "random", "misclassified"])
parser.add_argument("--output_dir", type=str, default="outputs/figures/gradcam")
```

---

## Phase IV — Sample selection policy

- [x] Reuse `evaluate.py` uncertainty CSV outputs to pick meaningful cases
- [x] Create four mandatory analysis buckets:
  1. Correct + low entropy
  2. Correct + high entropy
  3. Incorrect + high entropy
  4. Referable false negatives

### Selection snippet

```python
def select_case_ids(uncertainty_df: pd.DataFrame, n_per_bucket: int = 5) -> dict[str, list[str]]:
    df = uncertainty_df.copy()
    df["correct"] = df["true_grade"] == df["predicted_grade"]
    entropy_q75 = df["entropy"].quantile(0.75)
    entropy_q25 = df["entropy"].quantile(0.25)

    buckets = {
        "correct_low_entropy": df[df["correct"] & (df["entropy"] <= entropy_q25)],
        "correct_high_entropy": df[df["correct"] & (df["entropy"] >= entropy_q75)],
        "incorrect_high_entropy": df[(~df["correct"]) & (df["entropy"] >= entropy_q75)],
        "referable_false_negative": df[(df["true_grade"] >= 2) & (df["predicted_grade"] < 2)],
    }
    return {
        name: subset.head(n_per_bucket)["image_id"].astype(str).tolist()
        for name, subset in buckets.items()
    }
```

---

## Phase V — Thesis-ready outputs

- [x] Export figure panels per case
- [x] Export summary markdown/CSV with interpretability observations
- [x] Add side-by-side baseline vs CBAM overlays for identical image IDs

Expected outputs:

```text
outputs/figures/gradcam/
├── baseline_layer4_<image_id>.png
├── cbam_layer4_<image_id>.png
├── cbam_cbam4_<image_id>.png
└── gradcam_case_summary.csv
```

---

## 5) Clinical interpretation rubric

Use a consistent rubric when inspecting maps:

- **Plausible focus**: map overlaps vascular/lesion regions, avoids black border/background.
- **Failure pattern A**: optic disc fixation for non-disc pathology classes.
- **Failure pattern B**: peripheral border artifacts dominate saliency.
- **Failure pattern C**: diffuse non-informative saliency with no lesion concentration.

Include these categories in `gradcam_case_summary.csv`.

---

## 6) Validation checks

- [x] Determinism check: same image + same class produces same map across 2 runs.
- [x] Layer check: `layer4` and `layer3` maps are generated without hook failure.
- [x] Range check: CAM min/max in [0, 1].
- [x] Shape check: upsampled CAM matches image resolution (512x512).
- [x] Output check: required files written to `outputs/figures/gradcam/`.

---

## 7) Risks and mitigations

- **Risk**: Grad-CAM map too coarse at `layer4` (16x16).  
  **Mitigation**: add optional `layer3` target for finer maps.

- **Risk**: MC Dropout stochasticity makes maps unstable.  
  **Mitigation**: force deterministic mode during map generation.

- **Risk**: Heatmaps look convincing but are not clinically meaningful.  
  **Mitigation**: use predefined lesion plausibility rubric and false-negative-focused review.

---

## 8) Detailed TODO checklist

## Phase A — Core module

- [x] Create `src/interpretability.py`
- [x] Implement `GradCAM` class with hooks
- [x] Implement CAM normalization and upsampling
- [x] Add docstrings + type hints

## Phase B — Rendering utilities

- [x] Implement RGB de-normalization helper
- [x] Implement heatmap overlay helper
- [x] Implement panel plot function

## Phase C — Runner script

- [x] Create `src/run_gradcam.py`
- [x] Add checkpoint wildcard resolution (reuse `evaluate.py` pattern)
- [x] Add dataset/model/layer CLI arguments
- [x] Add save-path controls

## Phase D — Case selection and analysis

- [x] Parse uncertainty CSV from `outputs/results/`
- [x] Implement bucketed case selection
- [x] Generate maps for baseline + CBAM on same IDs
- [x] Export case summary CSV

## Phase E — Thesis integration

- [x] Add command examples to deployment docs
- [x] Add results interpretation template
- [x] Add figure naming convention for Chapter 4/5

---

## 9) Execution commands (planned)

Smoke test:

```bash
python src/run_gradcam.py \
  --checkpoint "outputs/checkpoints/baseline_resnet50*fold0_best.pth" \
  --model baseline \
  --dataset messidor2 \
  --target_layer layer4 \
  --num_images 5 \
  --selection random
```

CBAM high-entropy analysis:

```bash
python src/run_gradcam.py \
  --checkpoint "outputs/checkpoints/cbam_resnet50*fold0_best.pth" \
  --model cbam \
  --dataset messidor2 \
  --target_layer cbam4 \
  --num_images 20 \
  --selection high_entropy
```

---

## 10) Acceptance criteria

This plan is complete when:

1. Grad-CAM maps are generated for both baseline and CBAM on shared image IDs.
2. Outputs include overlays and structured case summary.
3. At least one figure set is ready for thesis inclusion.
4. Findings are integrated into discussion as evidence for clinical trust and failure-mode analysis.

