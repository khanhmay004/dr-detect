# Comprehensive Evaluation Report (Phase 3 Updated)

## 1) Executive Summary

This report is a full Phase 3 refresh of the previous Phase 2 evaluation, with outdated assumptions removed and current evidence integrated from end-to-end artifacts.

Overall conclusion:

- The project pipeline (preprocessing → training → external evaluation) is technically solid and thesis-viable.
- Phase 3A–3D completed successfully (calibration metrics + reliability + referral analysis + full reruns).
- Baseline remains stronger than current CBAM on core performance and referral utility.
- CBAM shows a small ECE improvement, but not enough to offset lower discrimination and sensitivity.
- Main unresolved clinical issue remains low referable sensitivity at default operating point.
- Decision to defer Phase 3E and redesign CBAM is reasonable and evidence-based.

---

## 2) Scope and Evidence Base

This analysis covers all work from project start through completed Phase 3 (A–D), using:

- Primary project status doc: `docs/research-project.md` (updated)
- Phase 3 plan/status: `plans/05-phase3.md` (updated)
- Phase 3 artifacts: `phase3-results/outputs/results/*` and `phase3-results/outputs/figures/*`
- Full Messidor-2 reruns (`T=20`, `n=1744`) for baseline and CBAM
- Referral-curve CSV outputs and calibration metrics (ECE/Brier)

Note: This document keeps the original filename for continuity, but it is now the **Phase 3 consolidated report**.

---

## 3) Progress Snapshot by Phase

## Phase 1 — Baseline foundation (complete)

What was achieved:

- Stable GPU training workflow established.
- Baseline fold-0 performance reached strong internal validation levels.
- Reproducible artifact generation pipeline established.

Why it matters:

- Baseline became a credible reference model, not a weak comparator.
- This raised the bar for CBAM and made neutral/negative ablation outcomes meaningful.

## Phase 2 — Baseline vs CBAM external comparison (complete)

What was achieved:

- Baseline and CBAM both evaluated on Messidor-2 under consistent protocol.
- External generalization behavior and class-level weaknesses became clear.

Why it matters:

- The project moved beyond in-domain metrics and tested domain shift robustness.
- This supports thesis rigor even when results are not uniformly positive.

## Phase 3A–3D — Calibration and referral analysis (complete)

What was achieved:

- Added ECE and Brier score to evaluation outputs.
- Added reliability diagram generation.
- Added referral-threshold/coverage analysis outputs.
- Re-ran full Messidor-2 evaluation for baseline and CBAM with new outputs.

Why it matters:

- Uncertainty claims are now measurable, not only qualitative.
- The thesis now has calibration evidence and operational referral trade-off data.

## Phase 3E — cross-fold CBAM expansion (deferred)

Status:

- Deferred intentionally (redesign-first path selected).

Reasoning:

- Current fold-0 + external evidence shows CBAM underperforming baseline.
- Running folds 1–4 before redesign is less efficient than first improving the architecture.

---

## 4) Final Quantitative Results (Locked for Current Cycle)

## 4.1 Messidor-2 full rerun results (Phase 3D)

Dataset/protocol:

- Dataset: Messidor-2 (gradable subset)
- Images: 1744
- MC passes: 20
- Task: 5-class DR grading + binary referable analysis (grade >= 2)

| Metric | Baseline | CBAM |
| --- | ---: | ---: |
| Accuracy | **0.6399** | 0.6095 |
| Quadratic Kappa | **0.6000** | 0.5784 |
| Binary Referable AUC | **0.8911** | 0.8778 |
| Referable Sensitivity | **0.4464** | 0.3720 |
| Referable Specificity | **0.9767** | 0.9736 |
| ECE | 0.1338 | **0.1201** |
| Brier Score | **0.5084** | 0.5170 |

Interpretation:

- Baseline wins on all major discrimination and referral metrics.
- CBAM is only better on ECE, but worse on Brier and clinical recall.
- The “CBAM helps” claim is not supported in current evidence.

## 4.2 Delta analysis (CBAM - Baseline)

- Accuracy: -0.0304
- QWK: -0.0216
- Referable AUC: -0.0134
- Referable Sensitivity: -0.0744
- Referable Specificity: -0.0031
- ECE: -0.0137 (improvement)
- Brier: +0.0086 (worse)

Interpretation:

- CBAM’s calibration gain is narrow and not consistent across calibration metrics.
- Clinically, the sensitivity drop is the most concerning change.

---

## 5) Class-Level Clinical Behavior Analysis

From the Phase 3 metrics JSON classification reports:

- Both models are strong on **No DR** recall (~0.96).
- Both models are weak on **Mild** recall (~0.05), reflecting difficult boundary/noise.
- **Moderate** recall is a key separator:
  - Baseline: 0.2363
  - CBAM: 0.0951

Clinical implication:

- Since grade >= 2 is referable, weak Moderate detection directly reduces screening sensitivity.
- This explains why referable sensitivity remains low, especially for CBAM.

---

## 6) Referral-Curve Analysis (Coverage vs Performance)

## 6.1 Baseline referral behavior

From `baseline_*_referral_curve.csv`:

- At 50% coverage: accuracy 0.7993, referable sensitivity 0.3393, specificity 0.9890
- At 80% coverage: accuracy 0.7068, referable sensitivity 0.4557, specificity 0.9888
- At 95% coverage: accuracy 0.6522, referable sensitivity 0.4311, specificity 0.9802

Interpretation:

- Selective prediction raises retained-set accuracy as expected.
- Sensitivity improves only modestly with stricter uncertainty filtering.
- Baseline remains high-specificity biased across operating points.

## 6.2 CBAM referral behavior

From `cbam_*_referral_curve.csv`:

- At 50% coverage: accuracy 0.7821, referable sensitivity 0.0189, specificity 1.0000
- At 80% coverage: accuracy 0.6946, referable sensitivity 0.2710, specificity 0.9915
- At 95% coverage: accuracy 0.6310, referable sensitivity 0.3645, specificity 0.9864

Interpretation:

- CBAM shows extreme conservatism at lower coverage (near-zero sensitivity at 50%).
- Across comparable coverage bands, baseline keeps better sensitivity and accuracy.
- Current CBAM is not suitable as-is for a screening-first referral strategy.

## 6.3 Clinical reading

- Both models currently optimize toward avoiding false positives (high specificity).
- For DR screening, false negatives are critical; therefore, threshold policy adjustment is necessary.
- Referral curves justify a next step focused on operating-point selection and calibration post-processing before any deployment-style claim.

---

## 7) Calibration Analysis (Phase 3 Addition)

What changed in Phase 3:

- Calibration moved from “pending” to measured (ECE + Brier + reliability figure outputs).

Observed pattern:

- CBAM has slightly lower ECE.
- Baseline has better Brier and better discriminative utility.

Why this matters:

- ECE alone is not enough to decide better clinical model quality.
- Brier and referral sensitivity indicate baseline remains more practical for current objective.
- Calibration and discrimination must be interpreted jointly.

---

## 8) End-to-End Quality Assessment (Preprocessing → Model → Result)

## 8.1 Data preprocessing quality

Assessment: strong and fit-for-purpose.

Reasoning:

- Ben Graham pipeline and normalization are correctly integrated.
- External dataset schema handling and gradable filtering are implemented.
- No evidence from Phase 3 suggests preprocessing is the bottleneck.

## 8.2 Model formulation quality

Assessment: methodologically strong, mixed empirical outcomes.

Reasoning:

- Baseline and CBAM were compared under consistent evaluator.
- MC Dropout uncertainty framework is functional.
- CBAM formulation in current design does not produce net gain.

## 8.3 Implementation quality

Assessment: good research engineering standard.

Reasoning:

- Evaluation scripts now produce complete uncertainty outputs.
- Artifact set includes JSON/CSV/figures needed for thesis reporting.
- Phase completion and documentation are now aligned.

## 8.4 Result quality relative to project goal

Assessment: good thesis quality, not clinically deployment-ready.

Reasoning:

- External QWK ~0.60 and referable AUC ~0.89 are useful for academic analysis.
- Referable sensitivity remains too low for strong screening claims.
- This is acceptable if discussed transparently as a limitation and active improvement area.

---

## 9) What Is Outdated (and Now Rewritten)

The previous version of this file contained now-outdated assumptions:

- “Calibration metrics missing” -> now completed (ECE/Brier/reliability outputs present).
- “Phase 3 pending” -> now updated to Phase 3A–3D complete.
- “Next immediate action is implementing calibration” -> now replaced by post-Phase-3 actions.

This report supersedes those points with current evidence.

---

## 10) Strategic Decision Review: Was Deferring Phase 3E Reasonable?

Short answer: yes, under your current objective.

Reasoning:

- You already obtained strong evidence that current CBAM underperforms baseline externally.
- Additional folds on a likely suboptimal design may consume substantial resources with limited insight.
- Redesign-first can produce a better candidate, then fold expansion can validate robustness on improved architecture.

Caveat:

- Keep clear in thesis that robustness claims are currently fold-limited for CBAM.

---

## 11) Recommended Next Steps (Post-Phase-3, Pre-Redesign and Redesign)

## 11.1 Immediate (before redesign experiments)

1. Define a baseline referral operating point (screening-first target).
2. Apply post-hoc temperature scaling and compare:
   - ECE
   - Brier
   - referable sensitivity/specificity at selected thresholds.
3. Lock baseline as final benchmark policy.

## 11.2 Redesign phase (CBAM)

1. Redesign CBAM placement/strength with a focused hypothesis.
2. Re-train fold 0 first, then external evaluation.
3. Only if redesigned CBAM shows clear gain, run fold 1–4 expansion.

## 11.3 Reporting phase

1. Prepare thesis tables with both discrimination and calibration metrics.
2. Explicitly discuss low-sensitivity risk and mitigation via referral policy.
3. Present CBAM underperformance as an honest, informative ablation result.

---

## 12) Final Verdict

Your project is in a strong state academically:

- Pipeline quality is high.
- Documentation and artifacts now support a full uncertainty-aware evaluation narrative.
- Baseline is robustly better than current CBAM in practical terms.

What this means:

- You can confidently present a high-quality thesis with transparent evidence.
- The next high-value move is **targeted CBAM redesign**, not blind continuation of current variant across more folds.

