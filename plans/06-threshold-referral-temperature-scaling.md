# Plan: Threshold/Referral Operating-Point Selection + Temperature Scaling

> **Version**: 1.0  
> **Date**: 2026-03-31  
> **Context**: Phase 3A–3D complete; baseline outperforms CBAM on core metrics; Phase 3E deferred for redesign-first path.

---

## 1) Goal

Define a clinically safer decision policy **without retraining** by:

1. Selecting threshold/referral operating points from existing uncertainty outputs.
2. Applying post-hoc temperature scaling to improve probability calibration.
3. Locking a final baseline policy to use as the benchmark before CBAM redesign.

---

## 2) Current Status (Why this plan now)

From completed Phase 3 reruns (Messidor-2, `T=20`, `n=1744`):

- Baseline: Acc 0.6399, QWK 0.6000, Ref-AUC 0.8911, Sens 0.4464, Spec 0.9767, ECE 0.1338, Brier 0.5084
- CBAM: Acc 0.6095, QWK 0.5784, Ref-AUC 0.8778, Sens 0.3720, Spec 0.9736, ECE 0.1201, Brier 0.5170

Key issue:

- Both models are specificity-heavy and sensitivity-light for referable DR.
- Calibration exists but needs post-hoc improvement and policy-level thresholding.

Decision:

- Use **baseline** as policy target model for this stage.

---

## 3) Target Outputs

Create a new result package under `outputs/results/` with:

1. `baseline_temp_scaling_fit.json` (learned temperature + fit diagnostics)
2. `baseline_calibrated_metrics.json` (ECE/Brier + discrimination metrics)
3. `baseline_operating_points.csv` (candidate thresholds and clinical trade-offs)
4. `baseline_operating_point_recommendation.md` (selected policy + rationale)
5. Figures:
   - reliability before/after scaling
   - sensitivity/specificity vs threshold
   - referral-rate vs sensitivity

---

## 4) Policy Definition

Use a two-stage screening policy:

1. **Referable classifier stage**: `P(referable)` thresholded from multiclass probs:
   - `P_ref = sum(P(class>=2))`
2. **Uncertainty gate stage**:
   - If entropy > threshold, auto-refer to human grader regardless of class.

Final decision logic:

```python
if entropy > entropy_threshold:
    decision = "REFER_HUMAN"
elif p_referable >= p_ref_threshold:
    decision = "REFER_DR"
else:
    decision = "NO_REFER"
```

This supports safer triage by catching uncertain cases.

---

## 5) Workstreams

## Workstream A — Data extraction for calibration/tuning

- Inputs:
  - Existing per-image outputs from evaluation (`mean_probs`, labels, entropy, predictions).
- If needed, extend evaluator to export logits/probs in a dedicated CSV/NPY for fitting.

### Planned command

```bash
python src/evaluate.py \
  --model baseline \
  --checkpoint "outputs/checkpoints/baseline_resnet50*fold0_best.pth" \
  --dataset messidor2 \
  --mc_passes 20 \
  --batch_size 16 \
  --save_probs
```

---

## Workstream B — Temperature scaling

Fit scalar temperature `T > 0` by minimizing NLL on validation-aligned outputs.

Scaled probabilities:

```python
scaled_logits = logits / T
scaled_probs = softmax(scaled_logits, axis=1)
```

Optimization objective:

```python
T_star = argmin_T NLL(softmax(logits / T), y_true)
```

Constraints:

- Fit temperature on validation-like set only (no test leakage if strict split available).
- Report both pre- and post-scaling ECE and Brier.

### Planned command

```bash
python src/calibrate_temperature.py \
  --input outputs/results/baseline_*_per_image_probs.csv \
  --output outputs/results/baseline_temp_scaling_fit.json
```

---

## Workstream C — Operating-point sweep

Sweep both:

- `p_ref_threshold` (e.g., 0.10 to 0.90 step 0.01)
- `entropy_threshold` (quantiles or absolute values)

For each operating point compute:

- referable sensitivity/specificity
- PPV/NPV
- referral rate (% sent to human)
- miss rate on referable DR

### Planned command

```bash
python src/select_operating_point.py \
  --input outputs/results/baseline_*_per_image_probs.csv \
  --temperature_file outputs/results/baseline_temp_scaling_fit.json \
  --target_min_sensitivity 0.85 \
  --output_csv outputs/results/baseline_operating_points.csv
```

---

## Workstream D — Selection criteria and recommendation

Primary criterion (screening-first):

- Choose operating points with `referable_sensitivity >= 0.85`.

Secondary criteria:

- Maximize specificity under sensitivity constraint.
- Keep referral rate operationally feasible.
- Prefer lower ECE/Brier post-scaling.

Deliver one recommended operating point and two alternates:

1. High-sensitivity policy (recommended)
2. Balanced policy
3. Low-referral policy

---

## 6) Detailed TODO Checklist

## Phase A — Preparation

- [ ] Confirm baseline artifact paths for latest full run
- [ ] Ensure per-image probabilities/logits are exported
- [ ] Freeze current baseline checkpoint/version for policy tuning

## Phase B — Temperature scaling implementation

- [ ] Create `src/calibrate_temperature.py`
- [ ] Implement scalar temperature optimization
- [ ] Save calibration fit metadata (`T`, NLL before/after, ECE/Brier before/after)
- [ ] Generate before/after reliability figures

## Phase C — Operating-point sweep implementation

- [ ] Create `src/select_operating_point.py`
- [ ] Implement 2D sweep (`p_ref_threshold`, `entropy_threshold`)
- [ ] Compute sensitivity/specificity/PPV/NPV/referral-rate/miss-rate
- [ ] Export full operating-point table CSV

## Phase D — Policy selection

- [ ] Select recommended threshold pair using screening-first criterion
- [ ] Write short rationale in `baseline_operating_point_recommendation.md`
- [ ] Validate chosen policy on full Messidor run artifacts

## Phase E — Documentation integration

- [ ] Update `docs/research-project.md` with calibrated policy summary
- [ ] Update `ideas/03-evaluate-phase2.md` with final operating-point conclusions
- [ ] Add runbook commands to Vast deployment doc if needed

---

## 7) Risks and Mitigations

- **Risk**: No threshold reaches target sensitivity with acceptable referral load.  
  **Mitigation**: prioritize safety (sensitivity), report operational trade-off explicitly.

- **Risk**: Temperature scaling improves ECE but worsens referral utility.  
  **Mitigation**: choose policy by clinical utility first; calibration is supporting evidence.

- **Risk**: Data leakage concerns in calibration fitting.  
  **Mitigation**: separate fit/eval partitions where possible; document limitations transparently.

---

## 8) Acceptance Criteria

Plan considered complete when:

1. Temperature scaling is fitted and reproducible from script.
2. Pre/post calibration metrics and reliability diagrams are generated.
3. Operating-point sweep table exists with clinical metrics.
4. One recommended baseline referral policy is selected and documented.
5. Results are integrated into project docs as pre-redesign benchmark.

