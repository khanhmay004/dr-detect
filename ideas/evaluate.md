# Evaluation: Uncertainty-Aware Attention CNN for DR Grading

> **Skill loaded**: `.agent/skills/research-and-paper/1-evaluate.md`
> **Evaluation date**: 2026-03-31 (updated after Phase 2 completion)
> **Evidence hierarchy**: T1 (DR-specific) → T2 (adjacent medical imaging) → T3 (general DL) → T4 (theoretical) → T5 (inference)

---

## 1. Thesis Viability Assessment — Bachelor's Level (Data Science)

### Verdict: **Supported — Strong Bachelor's Thesis**

The proposed thesis — CBAM-ResNet50 + MC Dropout + Focal Loss for 5-class DR grading — is well-scoped for a data science bachelor's thesis. It demonstrates:

| Requirement | Status | Evidence |
|---|---|---|
| Technical depth | ✅ | Custom attention module, Bayesian uncertainty via MC Dropout, loss function engineering |
| Novelty claim | ✅ | No single paper combines all three components in this configuration (see Section 2) |
| Empirical rigour | ✅ | 5-fold stratified CV, external validation on Messidor-2, multiple metrics (QWK, AUC, ECE) |
| Practical relevance | ✅ | Clinical screening use case, uncertainty-based referral threshold |
| Reproducibility | ✅ | Fixed seeds, public datasets, modular codebase |
| Scope management | ✅ | Explicit exclusions (no ViT, no lesion segmentation, no deployment) |

**What elevates this above a typical bachelor's thesis:**
- Cross-domain evaluation (APTOS → Messidor-2) — most student projects only validate internally
- Uncertainty quantification — moves beyond accuracy-chasing into clinical trustworthiness
- Ablation study design — isolates the contribution of each component

**What could make it fall short:**
- If only 1 fold is trained (currently only baseline fold 0 exists with batch_size=4 on CPU)
- If ECE/calibration metrics are omitted (see Gap 2 below)
- If the uncertainty analysis is superficial (just histograms, no referral threshold derivation)

---

## 2. Idea Evaluation (1-evaluate.md protocol)

### 2.1 Idea: CBAM-ResNet50 + MC Dropout + Focal Loss for 5-class DR grading

**Problem addressed**: Three simultaneous limitations of baseline ResNet-50 for DR:
1. Uniform spatial processing — no mechanism to attend to lesion regions
2. Point estimates only — no uncertainty quantification for clinical decision-making
3. Class imbalance bias — grade 0 dominates at ~49%, grade 3/4 are under-represented

---

### 2.2 Prior Art — Evidence Tiers

| Component | Paper | Evidence Tier | Result |
|---|---|---|---|
| CBAM for DR | MediDRNet (Teng 2024) | T1 — DR-specific | SOTA on minority DR classes using CBAM |
| CBAM for DR | IDANet (Bhati 2024) | T1 — DR-specific | AUC 0.86–0.96 across DDR/EyePACS/IDRiD |
| CBAM general | Woo et al. (ECCV 2018) | T3 — general DL | +1–2% accuracy on ImageNet, COCO |
| MC Dropout | Gal & Ghahramani (2016) | T3 — general DL | Bayesian approximation via dropout |
| MC Dropout for DR | Akram et al. (2025) | T1 — DR-specific | Bayesian approaches applied to DR grading |
| Focal Loss | Lin et al. (2017) | T3 — general DL | Addresses class imbalance in object detection |
| Focal Loss for DR | Romero-Oraá (2024) | T1 — DR-specific | Used with Xception backbone for DR grading |

**Key observation**: Each component individually has T1 support. No single paper combines all three in a unified framework. This gap is the thesis's novel contribution — it is genuine, not manufactured.

---

### 2.3 Mechanism Analysis

**CBAM** forces the network to weight lesion-containing spatial regions (spatial attention) and relevant feature channels (channel attention) more heavily. MediDRNet (T1) confirms CBAM specifically helps with subtle DR lesion features — microaneurysms, small haemorrhages. The mechanism is clinically grounded: DR lesions occupy a small fraction of the fundus image, and undirected convolutional processing wastes capacity on the healthy retinal background.

**MC Dropout** samples from an approximate posterior over network weights by keeping dropout active during inference. High variance across T=20 passes signals epistemic uncertainty — the model "knows what it doesn't know." This enables a human-in-the-loop referral pathway: high-entropy predictions are flagged for ophthalmologist review rather than auto-classified.

**Focal Loss** down-weights easy majority-class examples (grade 0, ~49%) by a factor of up to 100×, concentrating gradient signal on the rare but clinically critical minority classes (grades 3–4, ~13% combined). The alpha parameter further adjusts per-class weighting via inverse frequency.

---

### 2.4 Potential Failure Modes

#### Failure Mode 1 — Focal Loss × APTOS Label Noise Interaction
**Risk: MEDIUM** | Evidence: Chopra 2025 (T1)

Focal Loss amplifies gradients from hard examples. In APTOS 2019, some "hard" examples are hard because they are *mislabelled* (single grader, Kaggle crowdsource), not because they are clinically ambiguous. Focal Loss will over-fit to these noisy labels, particularly for the already-scarce grades 3 and 4.

> `[Evidence]` Chopra 2025, Section IX: "label protocol variability can cause large AUC swings."

**Mitigation**: Monitor per-class loss curves during training. If grade 3/4 loss diverges while grade 0/2 stabilises, label noise is amplifying. Consider label smoothing (ε=0.1) as a supplementary experiment.

#### Failure Mode 2 — Weak Bayesian Approximation from Single-Layer MC Dropout
**Risk: HIGH** | Evidence: `[Inference — no T1 evidence for this specific configuration]`

MC Dropout is applied only to the final Linear(2048→5) layer with p=0.5. This means:
- Dropout masks 1,024 of 2,048 features per pass on average
- But all 2,048 features are identically extracted by the deterministic backbone
- Variance across T=20 passes comes only from which half of the 2,048 features reach the 5 output neurons
- This is a very narrow uncertainty estimate — it captures uncertainty in the *decision boundary* but not in the *feature extraction*

In practice, this likely produces under-calibrated entropy: most predictions will have low entropy even when the model is genuinely uncertain, because the feature representation is identical across passes.

**Mitigation**: Two options:
1. **(Low effort)** Add MC Dropout before the avg-pool layer as well, so spatial features are also stochastically masked
2. **(Medium effort)** Report ECE (Expected Calibration Error) to quantify calibration quality — if ECE is high, the uncertainty estimates are not trustworthy

**Code snippet — Option 1 (add backbone-level dropout)**:

```python
# In model.py, CBAMResNet50.__init__:
self.backbone_dropout = MCDropout(p=0.1)  # lighter rate for backbone

# In CBAMResNet50.forward, after layer4 + cbam4:
x = self.layer4(x)
x = self.cbam4(x)
x = self.backbone_dropout(x)       # ← NEW: stochastic before pooling
x = self.avgpool(x)
x = torch.flatten(x, 1)
x = self.mc_dropout(x)             # existing head dropout
x = self.fc(x)
```

#### Failure Mode 3 — CBAM Spatial Attention Granularity at Deep Layers
**Risk: LOW-MEDIUM** | Evidence: `[Inference, grounded in IDRiD lesion annotations]`

At layer4, feature maps are 16×16. Each spatial attention cell covers 32×32 pixels in the 512×512 input. Microaneurysms are typically 10–15px in a 512×512 fundus image. Layer4 spatial attention is therefore too coarse to localise individual microaneurysms.

However, layer1 (128×128) and layer2 (64×64) have sufficient resolution. The multi-scale CBAM placement means early stages can attend to fine-grained lesions while deep stages capture global patterns (vascular distribution, disc/macula relationship).

**Assessment**: This is not a bug — it is an inherent property of the multi-scale architecture. Acknowledge it in the Discussion section: "CBAM at deeper layers captures global pathology patterns rather than individual lesion localisation."

#### Failure Mode 4 — Alpha Weights Computed from Full Dataset, Not Per-Fold
**Risk: LOW** | Evidence: Direct code inspection

Currently in `train.py` (line 344):
```python
train_labels = torch.tensor(train_df["diagnosis"].values)
alpha_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)
```

This correctly computes alpha from the *training fold*, not the full dataset. The train_df is already split at line 324:
```python
train_df, val_df = get_train_val_split(df, val_fold=args.fold)
```

**Status**: ✅ Already handled correctly. Alpha is fold-specific.

---

### 2.5 Dataset-Specific Notes

**APTOS 2019**:
- Known label noise (single Kaggle grader) → QWK is a more robust primary metric than accuracy
- Class imbalance: 49.3% grade 0, 5.3% grade 3 → justifies Focal Loss
- Stratified 5-fold CV is the correct strategy ✅
- Your project correctly prioritises QWK ✅

**Messidor-2**:
- Adjudicated labels from multiple ophthalmologists → cleaner ground truth
- Domain shift from APTOS (Indian hospitals, different cameras) → expect 5–15 QWK point drop
- 690 images (not 1,748 as stated in research.md — the 690 number comes from the filtered `adjudicated_dr_grade` subset)
- Your project correctly frames this as a domain generalization test ✅

---

### 2.6 Overall Verdict: **Supported**

Each component is T1-supported. The combination is novel. The failure modes are addressable. The scope is appropriate for a 1-month bachelor's thesis.

---

## 3. Literature Gaps Your Thesis Can Address

### Gap 1 — Calibration Reporting (ECE/Brier)
**Impact: HIGH** | **Implementation effort: LOW**

The Chopra 2025 survey explicitly calls out the absence of calibration metrics in DR papers. Your MC Dropout outputs are calibration-relevant (predictive entropy, confidence). Simply adding ECE computation to `evaluate.py` would close this gap.

**Current state**: `evaluate.py` does NOT compute ECE.

**Fix — add ECE to `evaluate.py`**:

```python
def compute_ece(
    mean_probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error — measures confidence-accuracy alignment.

    A perfectly calibrated model has ECE = 0: when it predicts 80% confidence,
    it is correct 80% of the time.

    Args:
        mean_probs: (N, C) predictive mean probabilities from MC Dropout.
        labels: (N,) ground-truth class indices.
        n_bins: Number of confidence bins.

    Returns:
        ECE as a float in [0, 1].

    Example:
        >>> probs = np.array([[0.9, 0.1], [0.6, 0.4]])
        >>> labels = np.array([0, 1])
        >>> compute_ece(probs, labels)  # low ECE if well-calibrated
    """
    confidences = mean_probs.max(axis=1)
    predictions = mean_probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece)
```

**Add reliability diagram plot**:

```python
def plot_reliability_diagram(
    mean_probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    save_path: Path | None = None,
) -> None:
    """Reliability diagram — visual calibration assessment.

    Perfect calibration = diagonal line. Below diagonal = overconfident.
    Above diagonal = underconfident.
    """
    confidences = mean_probs.max(axis=1)
    predictions = mean_probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_accs.append(accuracies[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Reliability plot
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_centers, bin_accs, width=1/n_bins, alpha=0.6,
            edgecolor="white", color="#3498db", label="Model")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Reliability Diagram", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of predictions per bin
    ax2.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.6,
            color="#2ecc71", edgecolor="white")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

### Gap 2 — Uncertainty-Aware Referral Thresholding
**Impact: HIGH** | **Implementation effort: MEDIUM**

No DR paper derives a specific entropy threshold for "refer to human grader" decisions and reports sensitivity/specificity at that threshold. Your thesis is uniquely positioned to close this gap.

**Implementation approach**:

```python
def compute_referral_curve(
    entropy: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold_percentiles: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute accuracy/coverage trade-off at different entropy thresholds.

    For each threshold t: predictions with entropy > t are 'referred'
    (flagged for human review). We report:
    - Coverage: fraction of images auto-classified (entropy <= t)
    - Accuracy: accuracy on auto-classified subset
    - Referred: fraction sent to human review

    Args:
        entropy: (N,) predictive entropy per image.
        labels: (N,) ground-truth grades.
        predictions: (N,) model predictions.
        threshold_percentiles: percentiles to evaluate (default: 5, 10, ..., 95).

    Returns:
        DataFrame with columns: threshold, coverage, accuracy, qwk, referred_pct
    """
    if threshold_percentiles is None:
        threshold_percentiles = np.arange(5, 100, 5)

    rows = []
    for pct in threshold_percentiles:
        threshold = np.percentile(entropy, pct)
        auto_mask = entropy <= threshold
        n_auto = auto_mask.sum()
        n_referred = (~auto_mask).sum()

        if n_auto > 0:
            auto_acc = (predictions[auto_mask] == labels[auto_mask]).mean()
            auto_kappa = cohen_kappa_score(
                labels[auto_mask], predictions[auto_mask], weights="quadratic"
            )
        else:
            auto_acc = 0.0
            auto_kappa = 0.0

        rows.append({
            "percentile": pct,
            "entropy_threshold": float(threshold),
            "coverage": float(n_auto / len(entropy)),
            "accuracy": float(auto_acc),
            "qwk": float(auto_kappa),
            "referred_pct": float(n_referred / len(entropy) * 100),
        })

    return pd.DataFrame(rows)
```

### Gap 3 — Cross-Dataset CBAM Ablation
**Impact: MEDIUM** | **Implementation effort: LOW (already designed)**

No paper ablates CBAM on/off across APTOS → Messidor-2. Your design (baseline ResNet-50 vs. CBAM-ResNet50, tested on both) directly addresses this. The baseline fold 0 is already trained; you just need the CBAM model fold 0 trained and both evaluated on Messidor-2.

---

## 4. Code-Level Issues and Fixes

### Issue 1 — Validation Uses MC Dropout (Unintended Stochasticity)
**Severity: MEDIUM** | **File**: `train.py`, line 137-139

```python
@torch.no_grad()
def validate(self, val_loader, criterion):
    """Validate with standard (non-MC) inference."""
    self.model.eval()
```

The docstring says "non-MC inference", but `MCDropout.forward()` hard-codes `training=True`. This means *every validation epoch* gets different results due to stochastic dropout. Validation loss and kappa metrics are non-deterministic during training, which introduces noise into:
- Early stopping decisions (kappa-based)
- Best model selection (kappa tracking)

**Impact**: The best model might not be the *actual* best model — it could be a lucky dropout mask sample.

**Fix approach**: Create a context manager that temporarily disables MC Dropout during training validation, while keeping it active during final MC inference.

```python
# In model.py, add to MCDropout:
class MCDropout(nn.Module):
    """Dropout layer that stays active during model.eval().

    Can be temporarily disabled via the mc_active flag for deterministic
    validation during training (as opposed to MC inference at test time).
    """

    def __init__(self, p: float = MC_DROPOUT_RATE):
        super().__init__()
        self.p = p
        self.mc_active = True  # ← NEW: toggle for training validation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mc_active:
            return F.dropout(x, self.p, training=True)
        return x


# Add helper to CBAMResNet50:
from contextlib import contextmanager

class CBAMResNet50(nn.Module):
    # ... existing code ...

    @contextmanager
    def deterministic_mode(self):
        """Temporarily disable MC Dropout for deterministic validation.

        Usage:
            with model.deterministic_mode():
                logits = model(images)  # no dropout stochasticity
        """
        # Find all MCDropout modules and disable them
        mc_modules = [m for m in self.modules() if isinstance(m, MCDropout)]
        for m in mc_modules:
            m.mc_active = False
        try:
            yield
        finally:
            for m in mc_modules:
                m.mc_active = True
```

```python
# In train.py, Trainer.validate():
@torch.no_grad()
def validate(self, val_loader, criterion):
    """Validate with deterministic (non-MC) inference."""
    self.model.eval()
    # ...
    for images, labels in pbar:
        # ...
        with self.model.deterministic_mode():  # ← NEW
            with torch.amp.autocast(...):
                logits = self.model(images)
                loss = criterion(logits, labels)
        # ...
```

### Issue 2 — `GaussNoise` Deprecated in Albumentations ≥ 1.4
**Severity: LOW** | **File**: `dataset.py`, line 220

```python
A.GaussNoise(var_limit=(10, 50), p=0.2),
```

In Albumentations ≥ 1.4, `GaussNoise` changed its API. The `var_limit` parameter now expects variance on a [0, 1] scale if the image is float, or [0, 255] if uint8. If you upgrade Albumentations, this line may silently produce wrong noise levels.

**Fix**: Pin the version in `requirements.txt` or update to the new API:

```python
# For Albumentations >= 1.4 with uint8 input:
A.GaussNoise(std_range=(0.02, 0.05), p=0.2),
```

### Issue 3 — Missing Gradient Clipping
**Severity: LOW** | **File**: `train.py`

Focal Loss with gamma=2.0 can produce large gradients on hard/noisy examples (Failure Mode 1). No gradient clipping is applied:

```python
# Current (train.py, lines 116-118):
self.scaler.scale(loss).backward()
self.scaler.step(optimizer)
self.scaler.update()
```

**Fix — add gradient clipping**:

```python
# train.py, after backward, before step:
self.scaler.scale(loss).backward()
self.scaler.unscale_(optimizer)                              # ← NEW
torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # ← NEW
self.scaler.step(optimizer)
self.scaler.update()
```

### Issue 4 — ECE Not Computed in evaluate.py
**Severity: HIGH (for thesis quality)** | **File**: `evaluate.py`

The `compute_metrics` function computes accuracy, QWK, and AUC — but not ECE or Brier score. Since your thesis's core claim involves uncertainty quantification, omitting calibration metrics is a critical gap.

**Fix**: Add the `compute_ece` function (provided in Section 3, Gap 1) and call it in `main()`. Add ECE to the metrics JSON output.

### Issue 5 — No Per-Patient Split Validation
**Severity: MEDIUM (for thesis defensibility)** | **File**: `dataset.py`

Voets et al. (2019) demonstrated that per-image splits (rather than per-patient) inflate performance metrics. Your APTOS split uses `StratifiedKFold` on images, not patients. APTOS 2019 does not provide patient IDs, so per-patient splitting is infeasible.

**Mitigation**: You cannot fix this with APTOS data. **Document it as a known limitation in the thesis.** For Messidor-2, your setup is correct because it's used purely as an unseen external test set (no split needed).

### Issue 6 — `EPOCHS` Used in Progress Bar Display Even With Custom Epochs
**Severity: LOW** | **File**: `train.py`, lines 99-101

```python
pbar = tqdm(
    train_loader,
    desc=f"Epoch {self.current_epoch + 1}/{EPOCHS} [Train]",
)
```

If `--epochs 30` is passed, the progress bar still shows `/20` because it references the config constant `EPOCHS` instead of the actual `num_epochs` parameter.

**Fix**: Pass `num_epochs` to `train_epoch` and `validate`, or store it as `self.num_epochs` in the Trainer.

```python
# In Trainer.__init__:
self.num_epochs = EPOCHS  # set in fit()

# In fit():
self.num_epochs = num_epochs

# In train_epoch / validate:
desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]"
```

### Issue 7 — Baseline Uses Different Architecture Than CBAM Model
**Severity: MEDIUM (for ablation validity)** | **File**: baseline notebook

The baseline checkpoint (`baseline_resnet50_fold0_best.pth`) was trained with `batch_size=4` on CPU for only 5 epochs. The CBAM model is designed for `batch_size=16` on GPU for 20 epochs. This means the ablation comparison (baseline vs. CBAM) is confounded by:

1. Different batch sizes (affects BatchNorm statistics and effective learning rate)
2. Different training duration (5 vs. 20 epochs — baseline may be undertrained)
3. Different compute (CPU vs. GPU — AMP not active on CPU)

**Fix**: Retrain the baseline ResNet-50 (without CBAM) using the **exact same** `train.py` pipeline on GPU with `batch_size=16`, `epochs=20`, and the same augmentation. The only difference should be: CBAM on vs. off.

```python
# Create a baseline model factory in model.py:
def create_baseline_model(
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = MC_DROPOUT_RATE,
    pretrained: bool = True,
) -> nn.Module:
    """Baseline ResNet-50 without CBAM, for ablation comparison.

    Uses the same head (MCDropout + Linear) as CBAMResNet50 to ensure
    the only variable is the attention mechanism.
    """
    backbone = models.resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    )
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        MCDropout(dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return backbone
```

---

## 5. What's Missing for a Complete Thesis (After Phase 2)

### 5.1 Critical Missing Pieces

| # | Missing Item | Impact | Current State |
|---|---|---|---|
| 1 | ECE + Brier calibration metrics | **Blocking** | Not implemented |
| 2 | Reliability diagram | **Blocking** | Not implemented |
| 3 | Referral threshold analysis | **Blocking** | Not implemented |
| 4 | CBAM folds 1–4 training | High | Not started (fold 0 complete) |
| 5 | CBAM cross-fold stats (mean ± std) | High | Not started |
| 6 | Ablation table A–D with calibrated metrics | High | Not started |
| 7 | Final publication-quality uncertainty figures | Medium | Partial |

### 5.2 Phase 2 Outcome Snapshot (Locked)

| Dataset | Metric | Baseline | CBAM | Delta (CBAM - Base) |
|---|---:|---:|---:|---:|
| APTOS val fold 0 | QWK | 0.9088 | 0.8896 | -0.0192 |
| APTOS val fold 0 | Accuracy | 0.8445 | 0.7899 | -0.0546 |
| APTOS val fold 0 | AUC | 0.9846 | 0.9831 | -0.0015 |
| Messidor-2 | Accuracy | 0.6399 | 0.6095 | -0.0304 |
| Messidor-2 | QWK | 0.6000 | 0.5777 | -0.0223 |
| Messidor-2 | Referable AUC | 0.8911 | 0.8778 | -0.0133 |
| Messidor-2 | Referable Sens | 0.4464 | 0.3720 | -0.0744 |
| Messidor-2 | Referable Spec | 0.9767 | 0.9736 | -0.0031 |

Interpretation: baseline currently outperforms CBAM in both internal fold-0 and external validation.

---

## 6. Detailed TODO List — Phase 3 Execution (Current)

### Phase 3A — Calibration Instrumentation

- [ ] Implement `compute_ece(mean_probs, labels, n_bins=15)` in `evaluate.py`
- [ ] Implement Brier score for multiclass (`np.mean(np.sum((p - y_onehot)^2, axis=1))`)
- [ ] Add ECE/Brier to output JSON
- [ ] Add reliability diagram plotting helper

### Phase 3B — Re-evaluation with Calibration

- [ ] Re-run Messidor-2 eval for baseline with calibration outputs
- [ ] Re-run Messidor-2 eval for CBAM with calibration outputs
- [ ] Build baseline-vs-CBAM table including ECE/Brier

### Phase 3C — Referral Policy Analysis

- [ ] Add entropy-threshold sweep (e.g., percentile thresholds 50/60/70/80/90/95)
- [ ] Compute coverage, accuracy-at-coverage, referable sensitivity, referable specificity
- [ ] Select candidate operating thresholds for “auto-accept vs refer-to-human”
- [ ] Save referral curve figures and CSV summaries

### Phase 3D — Robustness and Reporting

- [ ] Train CBAM folds 1–4
- [ ] Run `compute_cross_fold_stats.py` for CBAM (mean ± std, per-class metrics)
- [ ] Update thesis result tables with fold statistics
- [ ] Document negative/neutral CBAM result transparently if trend persists

---

## 7. Risk Assessment Summary (Post-Phase 2)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Calibration metrics remain missing | Medium | **Blocking** | Prioritize Phase 3A before new experiments |
| CBAM remains below baseline after more folds | Medium | High | Present as valid negative result; emphasize reproducibility and uncertainty |
| Low referable sensitivity remains unresolved | Medium | High | Use referral-threshold analysis, report screening-safe operating points |
| Time runs out before full CBAM folds | High | Medium | Minimum deliverable: calibrated fold-0 + external + referral analysis |

---

## 8. Final Recommendations — Priority Order (Now)

1. Add calibration metrics (ECE/Brier + reliability diagram) first.
2. Re-run baseline + CBAM Messidor with calibrated outputs.
3. Implement referral-threshold analysis and coverage-risk curves.
4. Train CBAM folds 1–4 for robust confidence intervals.
5. Finalize ablation and thesis tables with calibrated uncertainty reporting.
