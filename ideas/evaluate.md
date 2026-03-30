# Evaluation: Uncertainty-Aware Attention CNN for DR Grading

> **Skill loaded**: `.agent/skills/research-and-paper/1-evaluate.md`
> **Evaluation date**: 2026-03-30
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

## 5. What's Missing for a Complete Thesis

### 5.1 Critical Missing Pieces

| # | Missing Item | Impact | Current State |
|---|---|---|---|
| 1 | CBAM-ResNet50 training (any fold) | **Blocking** | Not started — only baseline exists |
| 2 | Messidor-2 evaluation | **Blocking** | Code exists, no checkpoint to evaluate |
| 3 | ECE / calibration metrics | High | Not implemented |
| 4 | Referral threshold analysis | High | Not implemented |
| 5 | Clean ablation (baseline retrained with same pipeline) | High | Baseline used different hyperparams |
| 6 | Confusion matrix + ROC visualization | Medium | Not generated |
| 7 | Training curves visualization | Medium | History JSON exists, no plots |

### 5.2 Nice-to-Have Additions

| # | Addition | Impact | Effort |
|---|---|---|---|
| A | Ordinal loss ablation (CORN / ordinal CE) | High — closes Gap 1 | Medium |
| B | Backbone-level MC Dropout | Medium — improves uncertainty | Low |
| C | Grad-CAM / attention map visualization | Medium — interpretability | Low |
| D | Label smoothing comparison | Low | Low |

---

## 6. Detailed TODO List — Phased Implementation Plan

### Phase 0: Infrastructure Fixes (Day 1)
> Fix code issues that affect all subsequent experiments.

- [ ] **0.1** Fix MCDropout deterministic validation
  - Modify `MCDropout` in `model.py` to support `mc_active` toggle (see Issue 1 fix)
  - Add `deterministic_mode()` context manager to `CBAMResNet50`
  - Update `Trainer.validate()` in `train.py` to use `deterministic_mode()`
  - Test: run 2 validation passes on same data → outputs must be identical

- [ ] **0.2** Add gradient clipping to training loop
  - Add `scaler.unscale_()` + `clip_grad_norm_(1.0)` in `train.py` (see Issue 3 fix)

- [ ] **0.3** Fix progress bar epoch display
  - Store `num_epochs` in Trainer, use it in tqdm descriptions (see Issue 6 fix)

- [ ] **0.4** Add baseline model factory
  - Create `create_baseline_model()` in `model.py` (see Issue 7 fix)
  - Ensure same head structure (MCDropout + Linear) as CBAM model

- [ ] **0.5** Add `--model` CLI argument to `train.py`
  - Support `--model baseline` and `--model cbam` to select architecture
  - Default to `cbam`

---

### Phase 1: Baseline Retraining (Day 2)
> Create a clean baseline with identical pipeline to CBAM model.

- [ ] **1.1** Retrain baseline ResNet-50 (fold 0)
  - Use `python train.py --model baseline --fold 0 --epochs 20`
  - Must use GPU with batch_size=16
  - Save checkpoint as `baseline_resnet50_fold0_best.pth`

- [ ] **1.2** Record baseline metrics
  - Val QWK, Val AUC, Val Accuracy after 20 epochs
  - Save training history JSON

- [ ] **1.3** Evaluate baseline on Messidor-2
  - Use `python evaluate.py --checkpoint baseline_best.pth`
  - Record: QWK, AUC, accuracy, entropy statistics

---

### Phase 2: CBAM-ResNet50 Training (Days 3–5)
> Train the proposed model and validate improvement over baseline.

- [ ] **2.1** Train CBAM-ResNet50 (fold 0)
  - `python train.py --model cbam --fold 0 --epochs 20`
  - Monitor training curves for overfitting (val_loss divergence)

- [ ] **2.2** Compare fold 0: baseline vs. CBAM
  - Create comparison table: QWK, AUC, accuracy
  - Check statistical significance (bootstrap CI or check if difference > noise)

- [ ] **2.3** Train CBAM-ResNet50 (folds 1–4)
  - Run all folds sequentially or parallel
  - Record per-fold metrics

- [ ] **2.4** Compute cross-fold statistics
  - Mean ± std for QWK, AUC, accuracy across 5 folds
  - This provides confidence intervals for the main results table

---

### Phase 3: Evaluation & Uncertainty Analysis (Days 6–8)
> External validation + uncertainty quantification.

- [ ] **3.1** Add ECE computation to `evaluate.py`
  - Implement `compute_ece()` function (see Section 3 code)
  - Add ECE to metrics JSON output
  - Add reliability diagram plot

- [ ] **3.2** Evaluate CBAM-ResNet50 on Messidor-2
  - MC Dropout inference with T=20 passes
  - Record: QWK, AUC, accuracy, ECE, mean entropy

- [ ] **3.3** Evaluate baseline on Messidor-2
  - Same pipeline, baseline checkpoint
  - Record same metrics for comparison

- [ ] **3.4** Generate uncertainty visualizations
  - Entropy histogram (colored by predicted grade) — already implemented ✅
  - Confidence vs. entropy scatter — already implemented ✅
  - Reliability diagram — to implement (see Section 3 code)
  - Per-class entropy box plot — to implement

- [ ] **3.5** Implement referral threshold analysis
  - `compute_referral_curve()` function (see Section 3 code)
  - Plot: coverage vs. accuracy at different entropy thresholds
  - Identify optimal threshold: best accuracy at ≥85% coverage
  - Report sensitivity/specificity at the optimal threshold

- [ ] **3.6** Case studies
  - Select 4–6 images from Messidor-2:
    - 2 correct + low entropy (model is right and confident)
    - 2 correct + high entropy (model is right but unsure)
    - 2 incorrect + high entropy (model is wrong and knows it)
  - For each: show fundus image, predicted distribution, entropy value
  - Discuss clinical implications

---

### Phase 4: Ablation Study (Days 9–10)
> Isolate the contribution of each component.

- [ ] **4.1** Design ablation table

  | Model | CBAM | Focal Loss | MC Dropout | QWK | AUC | ECE |
  |---|---|---|---|---|---|---|
  | A. ResNet-50 + CE | ✗ | ✗ | ✗ | — | — | — |
  | B. ResNet-50 + Focal | ✗ | ✓ | ✗ | — | — | — |
  | C. CBAM-ResNet50 + Focal | ✓ | ✓ | ✗ | — | — | — |
  | D. CBAM-ResNet50 + Focal + MC | ✓ | ✓ | ✓ | — | — | — |

  Each row adds exactly one component. The delta between adjacent rows isolates that component's contribution.

- [ ] **4.2** Train models A–C (fold 0 only — ablation on 1 fold is acceptable)
  - Model A: baseline + standard CE loss
  - Model B: baseline + Focal Loss
  - Model C: CBAM-ResNet50 + Focal Loss (deterministic eval, no MC at inference)
  - Model D: CBAM-ResNet50 + Focal Loss + MC Dropout (already from Phase 2)

- [ ] **4.3** Evaluate all 4 models on APTOS val fold 0 + Messidor-2
  - Fill in the ablation table
  - Write 1-paragraph interpretation per row delta

---

### Phase 5: Visualization & Figures (Days 11–12)
> Publication-quality figures for the thesis document.

- [ ] **5.1** Training curves
  - Loss (train/val) + QWK across epochs for baseline and CBAM
  - Overlay both models on the same plot

- [ ] **5.2** Confusion matrices
  - APTOS validation (baseline vs. CBAM) — side by side
  - Messidor-2 (baseline vs. CBAM) — side by side

- [ ] **5.3** ROC curves
  - Per-class one-vs-rest ROC for CBAM model
  - Binary referable ROC (grades ≥ 2) for both models

- [ ] **5.4** Uncertainty figures
  - Entropy histogram ✅ (exists)
  - Confidence vs. entropy scatter ✅ (exists)
  - Reliability diagram (to implement)
  - Referral coverage-accuracy curve (to implement)
  - Per-class entropy box plot (to implement)
  - Entropy: correct vs. incorrect predictions (violin plot)

- [ ] **5.5** Architecture diagram
  - Clean diagram of CBAM-ResNet50 for the thesis
  - Can be done in draw.io or tikz

---

### Phase 6: Additional Experiments (Days 13–14, if time permits)
> These close literature gaps and strengthen the thesis beyond minimum requirements.

- [ ] **6.1** Ordinal loss experiment (Gap 1)
  - Implement CORN loss or ordinal cross-entropy in `loss.py`
  - Train fold 0 with ordinal loss, compare QWK vs. Focal Loss
  - This is a clean, novel ablation

- [ ] **6.2** Backbone-level MC Dropout experiment
  - Add `MCDropout(p=0.1)` before avg-pool (see Failure Mode 2 fix)
  - Compare ECE: head-only dropout vs. backbone+head dropout
  - If ECE improves → report as a finding

- [ ] **6.3** Grad-CAM attention maps
  - Generate Grad-CAM heatmaps for selected Messidor-2 images
  - Compare CBAM model vs. baseline attention patterns
  - Show that CBAM focuses on lesion regions more than baseline

---

### Phase 7: Thesis Writing (Days 15–20)
> Parallel with final experiments.

- [ ] **7.1** Chapter 1 — Introduction
  - DR clinical background, AI motivation, research objectives
  - Cite: Alyoubi 2020 (historical), FDA meta-analysis 2025 (deployment bar)

- [ ] **7.2** Chapter 2 — Literature Review
  - ResNet (He 2015), CBAM (Woo 2018), MC Dropout (Gal 2016), Focal Loss (Lin 2017)
  - DR-specific: MediDRNet (Teng 2024), IDANet (Bhati 2024), RETFound (Zhou 2023)
  - Survey: Chopra 2025 ("From Retinal Pixels to Patients")
  - Reproducibility: Voets 2019

- [ ] **7.3** Chapter 3 — Methodology
  - Architecture diagram + component descriptions
  - Preprocessing pipeline (Ben Graham)
  - Training procedure (Focal Loss, AdamW, CosineAnnealing, AMP)
  - Evaluation protocol (5-fold CV, external validation, MC Dropout inference)

- [ ] **7.4** Chapter 4 — Results
  - Main results table (baseline vs. CBAM, APTOS + Messidor-2)
  - Ablation table
  - Uncertainty analysis results
  - All figures

- [ ] **7.5** Chapter 5 — Discussion
  - CBAM contribution analysis
  - Uncertainty calibration analysis (ECE)
  - Domain shift discussion (APTOS → Messidor-2)
  - Limitations:
    - Per-image not per-patient splits (APTOS limitation)
    - Single-layer MC Dropout may underestimate uncertainty
    - CBAM spatial attention too coarse at deep layers for MA localisation
    - Label noise in APTOS interacts with Focal Loss
  - Comparison with SOTA (IDANet, MediDRNet, RETFound)

- [ ] **7.6** Chapter 6 — Conclusion & Future Work
  - Summarise contributions
  - Future: ordinal losses, deeper MC Dropout, lesion segmentation (MTL), DeepDR Plus progression prediction

- [ ] **7.7** References & Appendix
  - Format all 30+ citations
  - Code appendix (key functions)

---

### Phase 8: Final Polish (Days 21–22)
> Code cleanup, documentation, and submission.

- [ ] **8.1** Clean codebase
  - Add/verify Google-style docstrings on all functions
  - Run `black` + `flake8` on all `.py` files
  - Remove unused imports

- [ ] **8.2** Write `README.md`
  - Reproduce instructions
  - Environment setup
  - Training + evaluation commands

- [ ] **8.3** Final thesis review
  - Proofread all chapters
  - Verify all figures referenced correctly
  - Check citation completeness

- [ ] **8.4** Submit

---

## 7. Risk Assessment Summary

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| GPU not available for full training | Medium | **Blocking** | Use Vast.ai ($5–7 total), or reduce to 1–2 folds |
| CBAM shows no improvement over baseline | Medium | High | Frame as a negative result with ablation analysis — still publishable |
| Messidor-2 performance very low (QWK < 0.5) | Medium | Medium | Expected due to domain shift — frame as discussion point, not failure |
| MC Dropout entropy poorly calibrated | High | Medium | Report ECE honestly; propose backbone-level dropout as fix |
| Time runs out before all 5 folds trained | High | Low | 1 fold + external validation is sufficient for bachelor's; note as limitation |
| Focal Loss overfits to APTOS label noise | Low | Medium | Monitor per-class loss; add label smoothing ablation if detected |

---

## 8. Final Recommendations — Priority Order

1. **Fix MC Dropout validation stochasticity** (Issue 1) — affects all training results
2. **Retrain baseline with identical pipeline** (Issue 7) — ablation is invalid without this
3. **Train CBAM-ResNet50 on GPU** — the thesis has no results without this
4. **Add ECE to evaluate.py** — closes the highest-impact literature gap with minimal code
5. **Implement referral threshold analysis** — this is the clinical contribution that distinguishes your thesis
6. **Run Messidor-2 evaluation** — external validation is what makes this thesis above-average
7. **Generate publication-quality figures** — visualisation is a significant portion of the thesis grade
8. **Write the thesis** — allocate at least 5 full days for writing, not just the last 2
