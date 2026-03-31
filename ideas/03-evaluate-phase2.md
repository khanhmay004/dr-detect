# Phase 2 Comprehensive Evaluation Report

## 1) Executive Summary

Phase 2 demonstrates that the project pipeline is already strong and thesis-viable, but the **current CBAM result (fold 0)** does **not** outperform the baseline on either internal validation or external Messidor-2. The baseline is currently the best-performing model.

Key conclusions:

- The end-to-end pipeline (preprocessing, training, MC inference, external testing) is functioning correctly and reproducibly.
- Baseline fold-0 performance is very strong on APTOS validation (**QWK 0.9088**, **Acc 84.45%**, **AUC 0.9846**).
- CBAM fold-0 underperforms baseline on APTOS validation (**QWK 0.8896**, **Acc 78.99%**, **AUC 0.9831**).
- On Messidor-2 external test, baseline again outperforms CBAM:
  - Baseline: **Acc 0.6399, QWK 0.6000, Ref-AUC 0.8911**
  - CBAM: **Acc 0.6095, QWK 0.5777, Ref-AUC 0.8778**
- Both models are clinically conservative (very high specificity, low sensitivity): good for avoiding false referrals, weak for catching all referable DR cases.
- Uncertainty behaves sensibly (incorrect predictions have higher entropy), but calibration quality (ECE/Brier) is still missing, which is a critical thesis gap.

Bottom line: your work from preprocessing to implementation is credible and technically solid; the current evidence supports a **strong baseline narrative** and a **negative/neutral CBAM ablation result** unless future folds or tuning reverse the trend.

---

## 2) What Was Evaluated

This assessment is based on:

- Project ground-truth documentation: `docs/research-project.md`
- Agent/rules context: `.agent/AGENTS.md`, `.agent/rules/*`
- Phase 2 artifacts under `phase2-results`, including:
  - Training/eval logs
  - Fold metrics JSONs
  - Cross-fold stats JSON
  - Messidor uncertainty CSVs
  - Generated figures/checkpoints
- Core implementation code:
  - `src/preprocessing.py`, `src/dataset.py`, `src/model.py`, `src/loss.py`, `src/train.py`, `src/evaluate.py`, `src/compute_cross_fold_stats.py`

---

## 3) Pipeline Quality Assessment (Start-to-End)

## 3.1 Data preprocessing

### What is good

- Ben Graham preprocessing is correctly implemented:
  - Circular crop + mask
  - Local color normalization (`4*I - 4*Blur + 128`)
- Implementation handles fallback for failed circle detection.
- Input is normalized to ImageNet statistics downstream.
- External dataset loading includes schema normalization and gradable filtering.

### Risks/notes

- APTOS lacks patient IDs, so per-image split remains a known validity limitation (already acknowledged in docs).
- Messidor CSV format compatibility is robust, but warnings are printed via `print` in dataset code; technically fine for this project, but less production-grade.

Assessment: **Strong and appropriate for thesis-level pipeline**.

---

## 3.2 Data augmentation and input strategy

### What is good

- Augmentation set is reasonable for fundus variability (flip/rotate/affine/brightness/noise).
- Train/val augmentation separation is clean.
- 512x512 preserves lesion-scale details better than low-resolution shortcuts.

### Risks/notes

- Training log shows warning:
  - `Argument(s) 'mode' are not valid for transform Affine`
- This indicates an API mismatch or obsolete argument usage path. It does not invalidate results but introduces avoidable noise and potential transform drift.

Assessment: **Good overall; minor technical hygiene issue remains**.

---

## 3.3 Model formulation and architecture choices

### What is good

- Controlled ablation intent is correct:
  - Same backbone/head/loss for baseline vs CBAM; only attention differs.
- MC Dropout design with deterministic validation context manager is a good engineering correction.
- Focal loss + class weights is justified by heavy class imbalance.

### Risks/notes

- MC Dropout is only at the classifier head, which may limit epistemic uncertainty richness (documented in your risk section as FM2).
- CBAM adds complexity but does not yet provide empirical gain in your current fold.

Assessment: **Methodologically sound design; CBAM value not empirically validated yet**.

---

## 3.4 Training implementation quality

### What is good

- AMP, AdamW, cosine schedule, gradient clipping, and early stopping are all aligned with modern best practice.
- Deterministic validation mode resolves earlier stochastic model-selection bug.
- Checkpoint naming/wildcard resolution in evaluation is a practical usability improvement.

### Important inconsistency found

- `baseline_resnet50_fold0_metrics.json` in `phase2-results/outputs/results` appears stale (5-epoch CPU-style metrics), while `baseline_resnet50_fold0_history.json` contains the strong 19-epoch GPU run.
- Correct baseline best metrics should be taken from the history/log and best checkpoint metadata:
  - **Epoch 14, QWK 0.908849, Acc 0.844475, AUC 0.984633**

Assessment: **Training implementation is good; artifact consistency needs cleanup for reporting integrity**.

---

## 3.5 Evaluation implementation quality

### What is good

- External evaluation runs full MC passes (`T=20`) on 1,744 gradable Messidor images.
- Outputs include both per-image uncertainty CSV and summary JSON.
- Binary referable metrics (AUC, sensitivity, specificity) are computed and reported.

### Critical gap

- Calibration metrics are still missing:
  - No ECE
  - No Brier score
  - No reliability diagram

Given the thesis claim around uncertainty/trustworthiness, this is the biggest methodological gap remaining.

Assessment: **Good external evaluation workflow, but incomplete uncertainty calibration story**.

---

## 4) Quantitative Results Analysis

## 4.1 Internal APTOS fold-0 (best epoch comparison)

Using the best available fold-0 runs:

- **Baseline (best from history):**
  - Epoch: 14
  - Val QWK: **0.9088**
  - Val Acc: **0.8445**
  - Val AUC: **0.9846**

- **CBAM (best from metrics):**
  - Epoch: 17
  - Val QWK: **0.8896**
  - Val Acc: **0.7899**
  - Val AUC: **0.9831**
  - Val Macro-F1: **0.6282**

### Interpretation

- CBAM is lower than baseline by:
  - QWK: ~**-0.0193**
  - Accuracy: ~**-0.0546**
  - AUC: ~**-0.0016**
- This is not a small random wiggle on accuracy; it is a material drop for fold-0.
- Baseline is already very strong, which makes incremental improvement harder.

---

## 4.2 External Messidor-2 comparison

### Baseline (20T, 1744 images)

- Accuracy: **0.6399**
- Quadratic Kappa: **0.6000**
- Referable AUC: **0.8911**
- Referable Sensitivity: **0.4464**
- Referable Specificity: **0.9767**

### CBAM (20T, 1744 images)

- Accuracy: **0.6095**
- Quadratic Kappa: **0.5777**
- Referable AUC: **0.8778**
- Referable Sensitivity: **0.3720**
- Referable Specificity: **0.9736**

### Delta (CBAM - Baseline)

- Accuracy: **-0.0304**
- QWK: **-0.0222**
- Referable AUC: **-0.0133**
- Sensitivity: **-0.0744**
- Specificity: **-0.0031**

### Interpretation

- Baseline generalizes better externally in current evidence.
- Both models favor non-referable predictions (high specificity, low sensitivity), which may be suboptimal for screening where missed referable DR is costly.
- External degradation from APTOS to Messidor is expected, but here CBAM does not improve robustness.

---

## 4.3 Class-level behavior (Messidor)

From classification reports:

- Strongest class performance is grade 0 (No DR) for both models.
- Mild and Moderate classes are weak, especially recall.
- CBAM severe/PDR behavior is mixed but does not compensate enough at macro level.

Clinical consequence:

- Model is good at identifying clear negatives.
- Model is weaker at detecting middle severity classes that often matter for referral boundaries.

---

## 4.4 Uncertainty behavior analysis (from uncertainty CSVs)

### Baseline uncertainty profile

- Mean entropy: **0.6512**
- Mean confidence: **0.7710**
- Entropy (correct vs incorrect): **0.5575 vs 0.8178**
- Confidence (correct vs incorrect): **0.8195 vs 0.6847**

### CBAM uncertainty profile

- Mean entropy: **0.7443**
- Mean confidence: **0.7296**
- Entropy (correct vs incorrect): **0.5959 vs 0.9759**
- Confidence (correct vs incorrect): **0.8049 vs 0.6120**

### Interpretation

- Good sign: both models produce higher entropy on errors.
- CBAM is globally more uncertain than baseline (higher entropy, lower confidence) while also being less accurate.
- This indicates uncertainty is informative, but CBAM currently shifts toward a less decisive and less accurate operating point.

---

## 4.5 Coverage-accuracy trade-off (entropy-based selective prediction)

When keeping only lower-entropy samples:

- Baseline at ~50% coverage: accuracy **~0.799**
- CBAM at ~50% coverage: accuracy **~0.782**
- At all tested coverage points (50/70/80/90/95%), baseline retains better selective accuracy.

Interpretation:

- Entropy can support a referral/defer strategy.
- But baseline still dominates CBAM under selective prediction in current fold-0 evidence.

---

## 5) Is Performance “Good” Overall?

Short answer: **Yes for baseline pipeline quality and thesis viability, mixed-to-weak for CBAM benefit claim (so far).**

### Why still good

- Baseline internal performance is excellent for this problem setting.
- External metrics are realistic and usable (QWK ~0.60, AUC-ref ~0.89), not collapsed.
- Uncertainty signal is meaningful.
- Engineering quality is high enough for reproducible thesis work.

### What is not yet good

- CBAM does not currently improve performance.
- Sensitivity for referable DR is low for both models.
- Calibration metrics are missing, which weakens the uncertainty claim.
- Cross-fold claim is currently effectively single-fold (`n_folds=1`), so statistical robustness is not yet demonstrated.

---

## 6) Root-Cause Hypotheses for CBAM Underperformance

Most plausible causes (in descending likelihood):

1. **Strong baseline ceiling effect**  
   Baseline already learns high-quality features on fold 0; CBAM extra flexibility may not help under this data regime.

2. **Small/noisy dataset + extra module complexity**  
   CBAM can overfit subtle patterns when labels are noisy and minority classes are scarce.

3. **Optimization dynamics mismatch**  
   Same LR/schedule may not be equally optimal for CBAM-inserted backbone.

4. **Attention placement granularity**  
   Stage-level CBAM (not inside each residual block) may be too coarse to consistently improve lesion localization.

5. **Evaluation threshold behavior for screening objective**  
   Model may need class-threshold tuning or cost-sensitive decision policy to improve sensitivity.

---

## 7) Thesis Framing Recommendation

Given current evidence, the strongest defensible framing is:

- “A strong reproducible baseline for DR grading with uncertainty quantification”
- “Attention (CBAM) does not automatically improve cross-domain performance under this setup”
- “Uncertainty remains clinically useful for selective referral, but calibration must be explicitly quantified”

This is still a valid and academically honest contribution, especially with transparent negative findings.

---

## 8) Priority Actions Before Final Thesis Claims

## Priority 1 (must-do)

1. **Add calibration metrics to `evaluate.py`**
   - ECE
   - Brier score
   - Reliability diagram

2. **Fix artifact consistency**
   - Ensure baseline fold-0 metrics JSON matches the real 19-epoch GPU run.
   - Remove/rename stale 5-epoch baseline metric artifact to avoid accidental misuse.

3. **Explicitly report model operating-point trade-off**
   - Highlight high specificity / low sensitivity.
   - Include threshold analysis for referable detection if possible.

## Priority 2 (strongly recommended)

4. **Run additional CBAM folds (or at least 2-3 folds)**
   - Current `n_folds=1` is insufficient for robust cross-fold conclusions.

5. **Perform lightweight tuning for CBAM**
   - Small LR sweep or warmup schedule
   - Possibly reduce dropout rate or adjust focal alpha sensitivity

## Priority 3 (if time allows)

6. **Selective prediction/referral protocol table**
   - Coverage vs accuracy and coverage vs sensitivity on referable DR.

7. **Error analysis on hard classes**
   - Focus on grade 1/2 confusion and false negatives in referable DR.

---

## 9) Detailed Strengths and Weaknesses Summary

### Strengths

- Clear clinical motivation and realistic problem framing
- Well-structured codebase and modular pipeline
- Correctly implemented deterministic-vs-MC dropout behavior
- External validation executed at full scale
- Uncertainty outputs available at per-image level for deeper analysis
- Baseline performance is genuinely strong

### Weaknesses / thesis risks

- CBAM benefit not demonstrated in current evidence
- Referable sensitivity is low
- Calibration metrics absent (core trustworthiness gap)
- Cross-fold statistics currently not meaningful (`n_folds=1`)
- Some artifact hygiene issues (stale baseline metrics JSON)

---

## 10) Final Verdict

You have built a solid and credible DR grading research pipeline from preprocessing through external evaluation. The baseline result is strong and publication-discussion worthy for a bachelor thesis.  

However, Phase 2 currently supports a **“strong baseline + non-improving CBAM ablation”** conclusion, not yet a “CBAM improves performance” conclusion. This is scientifically acceptable if reported transparently.

If you complete calibration reporting, clean metric artifacts, and add more CBAM folds, your final thesis quality will increase significantly and the uncertainty-aware contribution will be much more defensible.

