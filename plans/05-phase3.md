# Implementation Plan: Phase 3 — Calibration, Referral Policy, and Robust Evaluation

> **Document Version**: 1.0  
> **Created**: 2026-03-31  
> **Scope**: Post-Phase-2 execution plan to complete uncertainty credibility and thesis-grade evaluation artifacts.

---

## Executive Summary

Phase 2 established strong baseline performance and completed baseline-vs-CBAM fold-0 plus Messidor-2 external comparison.  

Phase 3 focuses on the highest-risk gap: **uncertainty calibration and decision policy validity**.

Primary outcomes for Phase 3:

1. Add **ECE + Brier + reliability diagram** to `src/evaluate.py`.
2. Add **referral-threshold analysis** (coverage/accuracy/sensitivity/specificity).
3. Re-run and report calibrated metrics for **baseline and CBAM** on Messidor-2.
4. Expand CBAM to folds 1–4 and compute robust cross-fold statistics.

---

## 1) Objectives and Success Criteria

## 1.1 Objectives

- Quantify whether uncertainty scores are calibrated (not just correlated with error).
- Define practical referral operating points for screening workflows.
- Improve statistical robustness beyond single-fold CBAM evidence.
- Produce publication/thesis-ready result tables and figures.

## 1.2 Success Criteria

- `evaluate.py` outputs ECE and Brier for every labeled run.
- Reliability diagram and referral curve are generated per run.
- Baseline and CBAM each have calibrated Messidor metrics.
- CBAM cross-fold stats file exists with mean ± std for key metrics.
- Phase 3 results can directly populate thesis Chapter 4 and Discussion.

---

## 2) Implementation Workstreams

## Workstream A — Calibration Metrics

### A1. Add ECE function

```python
def compute_ece(mean_probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = mean_probs.max(axis=1)
    predictions = mean_probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_idx in range(n_bins):
        in_bin = (confidences > bin_edges[bin_idx]) & (
            confidences <= bin_edges[bin_idx + 1]
        )
        bin_fraction = in_bin.mean()
        if bin_fraction == 0:
            continue
        bin_accuracy = accuracies[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += abs(bin_accuracy - bin_confidence) * bin_fraction
    return float(ece)
```

### A2. Add multiclass Brier score

```python
def compute_brier_score(mean_probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes, dtype=np.float32)[labels]
    return float(np.mean(np.sum((mean_probs - one_hot) ** 2, axis=1)))
```

### A3. Extend JSON metrics output

- Add fields:
  - `ece`
  - `brier_score`
  - `n_bins`

---

## Workstream B — Reliability Diagram

### B1. Add plotting helper

```python
def plot_reliability_diagram(
    mean_probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    save_path: Path,
) -> None:
    confidences = mean_probs.max(axis=1)
    predictions = mean_probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_acc = []
    for bin_idx in range(n_bins):
        in_bin = (confidences > bin_edges[bin_idx]) & (
            confidences <= bin_edges[bin_idx + 1]
        )
        if in_bin.any():
            bin_centers.append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
            bin_acc.append(accuracies[in_bin].mean())

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(bin_centers, bin_acc, marker="o", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
```

---

## Workstream C — Referral Threshold Analysis

### C1. Add coverage-risk sweep

```python
def compute_referral_curve(
    entropy: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    referable_threshold: int = 2,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    entropy_thresholds = np.quantile(entropy, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    for threshold in entropy_thresholds:
        keep_mask = entropy <= threshold
        coverage = float(keep_mask.mean())
        if keep_mask.sum() == 0:
            continue

        keep_labels = labels[keep_mask]
        keep_preds = predictions[keep_mask]
        accuracy = float((keep_labels == keep_preds).mean())

        keep_binary_labels = (keep_labels >= referable_threshold).astype(int)
        keep_binary_preds = (keep_preds >= referable_threshold).astype(int)

        tn = ((keep_binary_labels == 0) & (keep_binary_preds == 0)).sum()
        fp = ((keep_binary_labels == 0) & (keep_binary_preds == 1)).sum()
        fn = ((keep_binary_labels == 1) & (keep_binary_preds == 0)).sum()
        tp = ((keep_binary_labels == 1) & (keep_binary_preds == 1)).sum()

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        records.append(
            {
                "entropy_threshold": float(threshold),
                "coverage": coverage,
                "accuracy": accuracy,
                "referable_sensitivity": sensitivity,
                "referable_specificity": specificity,
            }
        )
    return pd.DataFrame(records).sort_values("coverage")
```

### C2. Add plots

- Coverage vs Accuracy
- Coverage vs Referable Sensitivity
- Coverage vs Referable Specificity

---

## Workstream D — Robustness Expansion (CBAM folds 1–4)

### D1. Training commands

```bash
python src/train.py --model cbam --fold 1 --epochs 20 --batch_size 16
python src/train.py --model cbam --fold 2 --epochs 20 --batch_size 16
python src/train.py --model cbam --fold 3 --epochs 20 --batch_size 16
python src/train.py --model cbam --fold 4 --epochs 20 --batch_size 16
```

### D2. Aggregate stats

```bash
python src/compute_cross_fold_stats.py --model cbam_resnet50 --use_metrics
```

---

## 3) Command Checklist (Execution Order)

1. Implement A + B + C in `src/evaluate.py`.
2. Smoke test on 10 images (`--max_images 10 --mc_passes 3`).
3. Full Messidor run for baseline (`--mc_passes 20`).
4. Full Messidor run for CBAM (`--mc_passes 20`).
5. Generate calibrated comparison table.
6. Run CBAM folds 1–4.
7. Compute cross-fold summary.
8. Update thesis result tables and discussion notes.

---

## 4) Detailed TODO (Phase + Task Granularity)

## Phase 3A — Calibration implementation

- [x] Add `compute_ece()` to `evaluate.py`
- [x] Add `compute_brier_score()` to `evaluate.py`
- [x] Include ECE/Brier in metrics JSON
- [x] Validate metrics on smoke run

## Phase 3B — Reliability visualization

- [x] Add reliability diagram helper
- [x] Save figure to `outputs/figures/*_reliability.png`
- [x] Verify plot exists for baseline + CBAM full runs

## Phase 3C — Referral policy analysis

- [x] Add referral curve computation
- [x] Save CSV to `outputs/results/*_referral_curve.csv`
- [x] Save coverage-performance plot
- [x] Select candidate operating points and summarize in table

## Phase 3D — Re-evaluation with calibrated outputs

- [ ] Re-run baseline Messidor (full)
- [ ] Re-run CBAM Messidor (full)
- [ ] Produce calibrated model comparison summary

## Phase 3E — Robustness extension

- [ ] Train CBAM fold 1
- [ ] Train CBAM fold 2
- [ ] Train CBAM fold 3
- [ ] Train CBAM fold 4
- [ ] Run cross-fold aggregation

## Phase 3F — Reporting integration

- [ ] Update `docs/research-project.md` metrics tables with calibrated outputs
- [ ] Update `ideas/evaluate.md` with final Phase 3 findings
- [ ] Prepare Chapter 4 results table and Chapter 5 discussion bullets

---

## 5) Risks and Mitigations

- **Risk**: ECE appears poor for both models.  
  **Mitigation**: report transparently, include temperature scaling as future work.

- **Risk**: CBAM still underperforms after 5 folds.  
  **Mitigation**: frame as negative result with proper statistical evidence.

- **Risk**: referral threshold improves accuracy but harms sensitivity too much.  
  **Mitigation**: present multiple operating points and recommend screening-safe threshold.

---

## 6) Deliverables at End of Phase 3

1. Updated `src/evaluate.py` with calibration + referral analysis.
2. Baseline and CBAM calibrated Messidor result JSON/CSV/figures.
3. CBAM cross-fold statistics file (mean ± std).
4. Updated documentation for thesis-ready reporting.

