# Research Project: Uncertainty-Aware Attention CNN for Diabetic Retinopathy Grading

> **Document Role**: Single source-of-truth for the `dr-detect` project.
> **Last Updated**: 2026-03-31
> **Project Type**: Bachelor's Thesis — Data Science (Deep Learning in Healthcare)
> **Timeline**: 1 month (01/03/2026 – 31/03/2026)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Clinical Background](#3-clinical-background)
4. [Research Objectives & Contributions](#4-research-objectives--contributions)
5. [Scope & Exclusions](#5-scope--exclusions)
6. [Literature Review & Prior Art](#6-literature-review--prior-art)
7. [Datasets](#7-datasets)
8. [Model Architecture](#8-model-architecture)
9. [Training Recipe](#9-training-recipe)
10. [Evaluation Protocol](#10-evaluation-protocol)
11. [Current Implementation Status](#11-current-implementation-status)
12. [Identified Issues & Fixes](#12-identified-issues--fixes)
13. [Literature Gaps & Novel Contributions](#13-literature-gaps--novel-contributions)
14. [Failure Modes & Risk Assessment](#14-failure-modes--risk-assessment)
15. [Phased Implementation Plan](#15-phased-implementation-plan)
16. [Expected Results](#16-expected-results)
17. [Thesis Structure](#17-thesis-structure)
18. [Future Work](#18-future-work)
19. [References](#19-references)

---

## 1. Project Overview

**Title**: _Uncertainty-Aware Attention CNN for Diabetic Retinopathy Grading from Fundus Photographs_

**Core Idea**: Combine three complementary techniques into a unified framework for 5-class DR grading:

| Component              | Technique                                   | Purpose                                                       |
| ---------------------- | ------------------------------------------- | ------------------------------------------------------------- |
| **Attention**          | CBAM (Convolutional Block Attention Module) | Focus on lesion regions instead of uniform spatial processing |
| **Uncertainty**        | MC Dropout (Monte Carlo Dropout)            | Quantify prediction confidence for clinical decision-making   |
| **Imbalance Handling** | Focal Loss with alpha weighting             | Concentrate gradient on rare but clinically critical classes  |

**Backbone**: ResNet-50 pre-trained on ImageNet (IMAGENET1K_V2 weights).

**Novelty Claim**: No single published paper combines all three components (CBAM + MC Dropout + Focal Loss) in this configuration for 5-class DR grading. Each component individually has Tier-1 (DR-specific) evidence support.

**Thesis Viability**: Assessed as a **strong bachelor's thesis** based on: technical depth (custom attention + Bayesian uncertainty + loss engineering), empirical rigour (5-fold stratified CV + external validation on Messidor-2), practical relevance (clinical screening use case with uncertainty-based referral), and reproducibility (fixed seeds, public datasets, modular codebase).

---

## 2. Problem Statement

### 2.1. Clinical Problem

Diabetic Retinopathy (DR) is the leading cause of preventable blindness in working-age adults (20–74 years). Approximately 1/3 of the 537 million people with diabetes worldwide develop DR signs. Early detection prevents >90% of severe vision loss cases, but manual screening by ophthalmologists is slow, expensive, and inaccessible in developing countries.

AI-based DR screening has reached clinical viability — 25+ FDA-cleared devices exist (Nature Digital Medicine meta-analysis 2025: pooled sensitivity 93%, specificity 90% across 82 studies / 887,244 examinations). The field has moved from research to deployment.

### 2.2. Technical Problem

Three simultaneous limitations of standard ResNet-50 for DR classification:

1. **Uniform spatial processing** — no mechanism to attend to lesion regions (microaneurysms, haemorrhages, exudates), which occupy a small fraction of the fundus image.
2. **Point estimates only** — standard models output a single prediction with no uncertainty quantification. A 51% confidence "No DR" prediction is clinically very different from a 99% prediction.
3. **Class imbalance bias** — DR datasets are heavily skewed toward "No DR" (49% in APTOS 2019), while clinically critical classes (Severe NPDR: 5.3%, Proliferative DR: 8.1%) are under-represented.

### 2.3. Research Gap

The field's current gap is not raw performance (Swin-L and TOViT push above 99% on APTOS) — it is **clinical trustworthiness**: calibrated uncertainty, reproducibility, and cross-domain generalization. This thesis addresses these gaps for a 2026 undergraduate thesis with a reproducible, uncertainty-aware pipeline with external validation and calibration metrics.

---

## 3. Clinical Background

### 3.1. DR Severity Scale (International Clinical DR Severity Scale)

| Grade | Name             | Description                            | Key Lesions                                         |
| ----- | ---------------- | -------------------------------------- | --------------------------------------------------- |
| 0     | No DR            | No retinal microvascular abnormalities | None                                                |
| 1     | Mild NPDR        | Microaneurysms only                    | Microaneurysms (MA) — small red dots, earliest sign |
| 2     | Moderate NPDR    | More than mild, less than severe       | MA + haemorrhages (HM) + hard exudates (EX)         |
| 3     | Severe NPDR      | Extensive abnormalities, ischemia      | Widespread HM, venous beading, IRMA                 |
| 4     | Proliferative DR | Neovascularization, highest risk       | New abnormal blood vessels, vitreous haemorrhage    |

**Referable DR threshold**: Grade ≥ 2 (Moderate or worse) → requires specialist referral.

### 3.2. Fundus Lesion Types

| Lesion              | Appearance                          | Clinical Significance           |
| ------------------- | ----------------------------------- | ------------------------------- |
| Microaneurysms (MA) | Small red dots (10–15px in 512×512) | Earliest DR sign                |
| Haemorrhages (HM)   | Dark red patches, larger than MA    | Vascular damage indicator       |
| Hard Exudates (EX)  | Bright yellow deposits              | Lipid leakage from vessels      |
| Cotton Wool Spots   | White fluffy patches                | Nerve fibre layer infarcts      |
| Neovascularization  | New abnormal blood vessels          | PDR marker, high blindness risk |

### 3.3. Why AI for DR Screening?

- Severe shortage of ophthalmologists in developing countries.
- AI achieves sensitivity ≥ 93% and specificity ≥ 90% (FDA meta-analysis 2025).
- Enables scalable, cost-effective screening at primary care level.
- Uncertainty quantification enables "human-in-the-loop" workflows where uncertain cases are referred to specialists.

---

## 4. Research Objectives & Contributions

### 4.1. Primary Objectives

1. Build a deep learning model for 5-class DR grading on fundus images.
2. Integrate CBAM attention to enhance lesion-relevant feature extraction.
3. Estimate prediction uncertainty via MC Dropout for clinical decision support.

### 4.2. Secondary Objectives

4. Evaluate cross-domain generalization on Messidor-2 (external dataset).
5. Compare baseline ResNet-50 vs. CBAM-ResNet50 via controlled ablation.
6. Analyse the relationship between uncertainty and prediction accuracy.

### 4.3. Expected Contributions

1. **Integrated architecture**: CBAM + MC Dropout + Focal Loss in a unified framework — few studies combine all three.
2. **Clinical applicability**: Per-prediction confidence scores enabling uncertainty-aware referral pathways.
3. **Cross-domain evaluation**: APTOS → Messidor-2 generalization assessment.
4. **Reproducible pipeline**: Fixed seeds, public datasets, modular codebase.
5. **Calibration reporting** (ECE/Brier): Addresses a gap explicitly called out by the Chopra 2025 survey.

---

## 5. Scope & Exclusions

### In Scope

- Image-level 5-class DR grading (classification).
- CNN-based architecture (ResNet-50 backbone).
- Uncertainty quantification via MC Dropout.
- 5-fold stratified cross-validation on APTOS 2019.
- External validation on Messidor-2.
- Ablation study isolating each component's contribution.

### Explicitly Excluded

- **No lesion segmentation** — image-level classification only.
- **No Vision Transformers** — focused on CNN with attention.
- **No deployment** — research pipeline only (no web/mobile app).
- **No federated learning** or self-supervised pre-training.
- **No ordinal loss functions** — potential future work (see Section 13).

---

## 6. Literature Review & Prior Art

### 6.1. Evidence Tiers

| Tier | Definition               | Example                                      |
| ---- | ------------------------ | -------------------------------------------- |
| T1   | DR-specific evidence     | MediDRNet uses CBAM for DR                   |
| T2   | Adjacent medical imaging | Skin lesion grading with ordinal loss        |
| T3   | General deep learning    | CBAM on ImageNet (Woo et al. 2018)           |
| T4   | Theoretical              | Bayesian approximation theory                |
| T5   | Inference                | No direct evidence, reasoned from principles |

### 6.2. Component-Level Prior Art

| Component         | Paper                   | Tier | Key Result                             |
| ----------------- | ----------------------- | ---- | -------------------------------------- |
| CBAM for DR       | MediDRNet (Teng 2024)   | T1   | SOTA on minority DR classes using CBAM |
| CBAM for DR       | IDANet (Bhati 2024)     | T1   | AUC 0.86–0.96 across DDR/EyePACS/IDRiD |
| CBAM general      | Woo et al. (ECCV 2018)  | T3   | +1–2% accuracy on ImageNet, COCO       |
| MC Dropout        | Gal & Ghahramani (2016) | T3   | Bayesian approximation via dropout     |
| MC Dropout for DR | Akram et al. (2025)     | T1   | Bayesian approaches applied to DR      |
| Focal Loss        | Lin et al. (2017)       | T3   | Addresses class imbalance in detection |
| Focal Loss for DR | Romero-Oraá (2024)      | T1   | Used with Xception for DR grading      |

**Key observation**: Each component individually has T1 support. No single paper combines all three — this gap is the thesis's novel contribution.

### 6.3. Architecture Evolution in DR (2016–2025)

| Era       | Dominant Approach                           | Examples                                        |
| --------- | ------------------------------------------- | ----------------------------------------------- |
| 2016–2021 | CNN backbones + ImageNet pre-training       | Inception-v3 (Gulshan), ResNet-50, EfficientNet |
| 2022–2024 | Hybrid (CNN + attention modules)            | CBAM-CNN (MediDRNet), IDANet, dual attention    |
| 2025+     | Full ViT, foundation models, neuro-symbolic | Swin-L, TOViT, RETFound (ViT-L/16)              |

This project sits in the **2022–2024 hybrid tier** — well-supported, practically relevant, appropriate scope for a bachelor's thesis.

### 6.4. Key Benchmark Results from Literature

| Paper            | Dataset    | Task    | AUC   | Accuracy | Notes                                  |
| ---------------- | ---------- | ------- | ----- | -------- | -------------------------------------- |
| Gulshan 2016     | EyePACS    | Binary  | 0.991 | —        | Private data, not reproducible         |
| Voets 2019       | Messidor-2 | Binary  | 0.85  | —        | Reproduction study — 14-point AUC drop |
| RETFound 2023    | APTOS      | 5-class | 0.943 | —        | Foundation model, 307M params          |
| IDANet 2024      | DDR        | 5-class | 0.86  | 0.85     | Dual attention, most direct comparator |
| Romero-Oraá 2024 | Kaggle     | 5-class | —     | 83.7%    | QWK 0.78, Xception + Focal Loss        |
| Swin-L 2025      | BRSET      | Binary  | 0.98  | 0.95     | 197M params, 8× larger than ResNet-50  |

### 6.5. Reproducibility Concerns

Voets et al. (2019) demonstrated that reproducing Gulshan et al.'s results with public data yielded AUC drops of 4–14 points due to: (1) private curation steps, (2) per-image rather than per-patient splits, (3) label adjudication differences. This is the canonical argument for transparent evaluation protocols.

---

## 7. Datasets

### 7.1. APTOS 2019 Blindness Detection (Training/Validation)

| Property       | Value                              |
| -------------- | ---------------------------------- |
| Source         | Kaggle competition                 |
| Total images   | 3,662                              |
| Classes        | 5 (grades 0–4)                     |
| Format         | PNG                                |
| Split strategy | 5-fold Stratified K-Fold (seed=42) |
| Label quality  | Single Kaggle grader — known noise |

**Class distribution (severe imbalance)**:

| Grade | Name          | Count | Percentage |
| ----- | ------------- | ----- | ---------- |
| 0     | No DR         | 1,805 | 49.3%      |
| 1     | Mild          | 370   | 10.1%      |
| 2     | Moderate      | 999   | 27.3%      |
| 3     | Severe        | 193   | 5.3%       |
| 4     | Proliferative | 295   | 8.1%       |

**Imbalance ratio**: 9.4:1 (grade 0 vs. grade 3). Justifies Focal Loss.

**Known limitation**: No patient IDs available → per-image splits only (not per-patient). Documented as a known limitation per Voets et al. (2019).

### 7.2. Messidor-2 (External Test Set)

| Property      | Value                                                                      |
| ------------- | -------------------------------------------------------------------------- |
| Source        | Decencière et al. (images) + Krause et al. 2018 (labels)                   |
| Total images  | 1,748 (874 examinations × 2 eyes)                                          |
| Gradable      | 1,745 (after filtering `adjudicated_gradable == 1`; 3 ungradable excluded) |
| Format        | PNG / JPG                                                                  |
| Role          | External domain generalization test                                        |
| Label quality | Adjudicated by 3 US board-certified retina specialists (ICDR 0–4 scale)    |
| Label source  | [Kaggle google-brain/messidor2-dr-grades](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) |

**Why Messidor-2**: Completely different domain from APTOS (French hospitals vs. Indian hospitals, different cameras, different patient demographics). Expected 5–15 QWK point drop due to domain shift — this is normal and a key discussion point.

**Label provenance note**: The official Messidor-2 release does not include DR severity grades. This project uses adjudicated labels from Krause et al. (2018), where each image was independently graded by three US board-certified retina specialists and adjudicated to consensus. This is the standard label source used by Gulshan et al. (2016), Voets et al. (2019), and subsequent literature.

### 7.3. Data Split Strategy

```
APTOS 2019 (3,662 images)
    │
    ├── 5-Fold Stratified K-Fold (seed=42)
    │   ├── Fold 0: Train ~2,929 | Val ~733
    │   ├── Fold 1–4: Same 80/20 split
    │   └── Stratified: class distribution preserved per fold
    │
Messidor-2 (1,745 gradable images) → External Test Set (never used for training)
```

---

## 8. Model Architecture

### 8.1. Proposed Model: CBAM-ResNet50

```
Input: Fundus image (B, 3, 512, 512)
         │
    ┌────▼──────────────────────────┐
    │ ResNet-50 Stem                │ → (B, 64, 128, 128)
    │  Conv7×7 → BN → ReLU → MaxPool│
    └────┬──────────────────────────┘
         │
    ┌────▼────┐
    │ Layer1  │ → 3 Bottleneck → (B, 256, 128, 128)
    │ + CBAM  │   Channel Att → Spatial Att
    └────┬────┘
         │
    ┌────▼────┐
    │ Layer2  │ → 4 Bottleneck → (B, 512, 64, 64)
    │ + CBAM  │
    └────┬────┘
         │
    ┌────▼────┐
    │ Layer3  │ → 6 Bottleneck → (B, 1024, 32, 32)
    │ + CBAM  │
    └────┬────┘
         │
    ┌────▼────┐
    │ Layer4  │ → 3 Bottleneck → (B, 2048, 16, 16)
    │ + CBAM  │
    └────┬────┘
         │
    ┌────▼──────────────────────────┐
    │ Adaptive Average Pool         │ → (B, 2048)
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │ MC Dropout (p=0.5)            │ ← Always ON (even in eval())
    └────┬──────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │ Linear(2048 → 5)              │ → (B, 5) logits
    └───────────────────────────────┘
```

### 8.2. Parameter Count

| Component          | Parameters |
| ------------------ | ---------- |
| ResNet-50 backbone | ~23.5M     |
| 4× CBAM blocks     | ~0.3M      |
| FC head (2048→5)   | ~10K       |
| **Total**          | **~23.8M** |

### 8.3. CBAM Module Details

**Channel Attention**: Identifies _which feature channels_ are important (e.g., lesion-detecting channels vs. background channels).

```
F ∈ R^(C×H×W) → AvgPool + MaxPool → Shared MLP (C→C/16→C) → Sigmoid → Scale F
```

**Spatial Attention**: Identifies _which spatial regions_ contain lesions.

```
F' → AvgPool_channel + MaxPool_channel → Concat → Conv7×7 → Sigmoid → Scale F'
```

**Design decision**: CBAM placed _after_ each ResNet stage (not inside residual blocks) to maintain ImageNet pre-trained weight compatibility.

### 8.4. MC Dropout Implementation

Custom `MCDropout(nn.Module)` that hard-codes `training=True` in `F.dropout()`, keeping dropout active even in `model.eval()`. This allows:

- BatchNorm to use running statistics (stable predictions)
- Dropout to sample from approximate Bayesian posterior (uncertainty)

Features an `mc_active` toggle and `deterministic_mode()` context manager for deterministic validation during training.

### 8.5. Baseline Model: BaselineResNet50

Identical to CBAMResNet50 but **without CBAM modules**. Uses the same classification head (MCDropout + Linear) to ensure the only variable in ablation is the attention mechanism. ~23.5M parameters.

### 8.6. Comparison: Baseline vs. Proposed

| Feature     | Baseline ResNet-50    | CBAM-ResNet50 (Proposed) |
| ----------- | --------------------- | ------------------------ |
| Attention   | None                  | CBAM at 4 stages         |
| Dropout     | MCDropout (always on) | MCDropout (always on)    |
| Uncertainty | Via MC Dropout        | Via MC Dropout           |
| Loss        | Focal Loss            | Focal Loss               |
| Parameters  | ~23.5M                | ~23.8M                   |

---

## 9. Training Recipe

### 9.1. Preprocessing: Ben Graham's Pipeline

**Step 1 — Circular Crop**: Detect retina boundary via thresholding + contour detection, crop with 5% margin, apply circular mask, resize to 512×512.

**Step 2 — Local Color Normalization**: `result = 4 × image − 4 × GaussianBlur(image, σ) + 128` where `σ = image_width / 30`. Removes illumination variation while preserving lesion detail. Processes all 3 BGR channels simultaneously (unlike CLAHE which processes per-channel).

### 9.2. Data Augmentation (Albumentations)

**Training**:

| Augmentation                     | Parameters                       | Purpose                |
| -------------------------------- | -------------------------------- | ---------------------- |
| HorizontalFlip                   | p=0.5                            | Spatial invariance     |
| VerticalFlip                     | p=0.5                            | Spatial invariance     |
| RandomRotate90                   | p=0.5                            | Rotational invariance  |
| ShiftScaleRotate                 | shift=0.1, scale=0.1, rotate=45° | Geometric perturbation |
| RandomBrightnessContrast / CLAHE | p=0.3 (OneOf)                    | Photometric variation  |
| GaussNoise                       | var_limit=(10, 50), p=0.2        | Noise robustness       |
| Normalize                        | ImageNet mean/std                | Pixel standardization  |

**Validation/Inference**: Resize(512, 512) + Normalize(ImageNet mean/std) only.

### 9.3. Hyperparameters

| Parameter         | Value                          | Rationale                                   |
| ----------------- | ------------------------------ | ------------------------------------------- |
| Image Size        | 512×512                        | Preserves MA detail; fits in VRAM with AMP  |
| Batch Size        | 16                             | Maximum for GPU 12GB with AMP               |
| Epochs            | 20                             | Sufficient with early stopping              |
| Learning Rate     | 1e-4                           | Low LR for fine-tuning pre-trained backbone |
| Optimizer         | AdamW                          | Decoupled weight decay                      |
| Weight Decay      | 1e-4                           | Regularization                              |
| Scheduler         | CosineAnnealingLR (T_max=20)   | Smooth LR decay                             |
| Focal Loss γ      | 2.0                            | Standard focusing parameter                 |
| Focal Loss α      | Auto (inverse class frequency) | Per-fold computation                        |
| MC Dropout p      | 0.5                            | Bayesian head dropout rate                  |
| MC Passes T       | 20                             | Inference stochastic passes                 |
| Early Stopping    | Patience=5 (on val QWK)        | Prevent overfitting                         |
| Gradient Clipping | max_norm=1.0                   | Stabilize Focal Loss gradients              |
| Random Seed       | 42                             | Reproducibility                             |
| AMP               | Enabled                        | Halves VRAM usage                           |

### 9.4. Focal Loss

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

- `γ = 2.0`: Down-weights easy/well-classified examples by up to 100×.
- `α_t`: Inverse-frequency class weights, computed per fold from training labels.

### 9.5. Training Infrastructure

- **Local**: CPU smoke tests (batch_size=2, epochs=2, image_size=256).
- **GPU**: Vast.ai cloud (RTX 3060/3070/4070, $0.15–$0.30/hr).
- **Estimated training time**: ~3–5 hours per fold on RTX 3060/3070.
- **Estimated total cost**: ~$5–7 for all 5 folds.

---

## 10. Evaluation Protocol

### 10.1. Metrics

| Metric                     | Abbreviation | Purpose                                                       |
| -------------------------- | ------------ | ------------------------------------------------------------- |
| Quadratic Weighted Kappa   | QWK          | **Primary metric** — ordinal agreement, robust to label noise |
| Accuracy                   | Acc          | Overall correctness                                           |
| Macro F1-Score             | F1           | Balanced per-class performance                                |
| AUC-ROC (one-vs-rest)      | AUC          | Per-class discrimination                                      |
| Binary Referable AUC       | AUC-Ref      | Grades ≥ 2 vs. < 2                                            |
| Sensitivity / Specificity  | Sen / Spec   | Clinical screening performance                                |
| Expected Calibration Error | ECE          | Confidence-accuracy alignment                                 |
| Predictive Entropy         | H            | Uncertainty quantification                                    |
| Confidence                 | Conf         | Max softmax probability                                       |

### 10.2. MC Dropout Inference

```
For each image x, perform T=20 stochastic forward passes:
  ŷ₁, ŷ₂, ..., ŷ_T  (softmax probability vectors)

Predictive mean:     ȳ = (1/T) Σ ŷ_t
Predicted class:     argmax(ȳ)
Predictive entropy:  H(ȳ) = -Σ ȳ_c · log(ȳ_c)
Confidence:          max(ȳ)
```

- H ≈ 0 → Very certain (all T passes agree)
- H > 1.0 → Very uncertain → flag for human review

### 10.3. Evaluation Workflow

1. **APTOS Validation**: Per-fold metrics (QWK, AUC, Acc) during training.
2. **Cross-Fold Statistics**: Mean ± std across 5 folds → confidence intervals.
3. **Messidor-2 External**: MC Dropout inference (T=20), full metrics + uncertainty analysis.
4. **Ablation Study**: 4-model comparison isolating each component.

### 10.4. Ablation Design

| Model                         | CBAM | Focal Loss | MC Dropout | QWK | AUC | ECE |
| ----------------------------- | ---- | ---------- | ---------- | --- | --- | --- |
| A. ResNet-50 + CE             | ✗    | ✗          | ✗          | —   | —   | —   |
| B. ResNet-50 + Focal          | ✗    | ✓          | ✗          | —   | —   | —   |
| C. CBAM-ResNet50 + Focal      | ✓    | ✓          | ✗          | —   | —   | —   |
| D. CBAM-ResNet50 + Focal + MC | ✓    | ✓          | ✓          | —   | —   | —   |

Each row adds exactly one component. Delta between adjacent rows isolates that component's contribution.

### 10.5. Visualizations Required

- Training/validation loss and QWK curves (baseline vs. CBAM overlay)
- Confusion matrices (APTOS val + Messidor-2, side-by-side)
- ROC curves (per-class one-vs-rest + binary referable)
- Entropy histogram (colored by DR grade)
- Confidence vs. entropy scatter plot
- Reliability diagram (calibration assessment)
- Referral coverage-accuracy curve
- Per-class entropy box plot
- Case studies (high/low uncertainty examples)
- Architecture diagram

---

## 11. Current Implementation Status

### 11.1. Codebase Structure

```
src/
├── config.py           # Centralized hyperparameters, paths, constants
├── preprocessing.py    # Ben Graham's pipeline (crop + color norm)
├── dataset.py          # DRDataset, MessidorDataset, augmentations, dataloaders
├── model.py            # CBAM, MCDropout, CBAMResNet50, BaselineResNet50
├── loss.py             # FocalLoss, compute_class_weights
├── train.py            # Trainer class (AMP, early stopping, checkpointing)
├── evaluate.py         # MC Dropout inference, metrics, visualization
├── configs/            # YAML experiment configs (CPU smoke test, GPU baseline, GPU CBAM)
│   ├── experiment_config.py
│   ├── cpu_smoke_test.yaml
│   ├── gpu_baseline.yaml
│   └── gpu_cbam.yaml
└── tests/              # Unit tests for model and determinism
```

### 11.2. Completed Work (Phase 0 + Phase 1)

**Phase 0: Infrastructure Fixes** (✅ Complete)

| Task                                                  | Status      | Date       |
| ----------------------------------------------------- | ----------- | ---------- |
| MCDropout `mc_active` toggle + `deterministic_mode()` | ✅ Complete | 2026-03-30 |
| Gradient clipping (`GRAD_CLIP_NORM = 1.0`)            | ✅ Complete | 2026-03-30 |
| Progress bar epoch display fix                        | ✅ Complete | 2026-03-30 |
| BaselineResNet50 model factory                        | ✅ Complete | 2026-03-30 |
| `--model {baseline, cbam}` CLI argument               | ✅ Complete | 2026-03-30 |
| Experiment config system (YAML)                       | ✅ Complete | 2026-03-30 |
| CPU smoke tests (baseline + CBAM, 2 epochs)           | ✅ Complete | 2026-03-30 |
| Validation determinism verification                   | ✅ Complete | 2026-03-30 |

**Phase 1: GPU Baseline Training** (✅ Complete — 2026-03-31)

| Task                              | Status      | Details                                            |
| --------------------------------- | ----------- | -------------------------------------------------- |
| GPU infrastructure setup          | ✅ Complete | NVIDIA RTX 3090 (25.3GB VRAM), Vast.ai             |
| Baseline ResNet-50 training       | ✅ Complete | Fold 0, 19 epochs, ~2.5 hours                      |
| Training artifacts                | ✅ Complete | Checkpoints (270MB × 2), logs, metrics, 19 figures |
| Model performance validation      | ✅ Complete | Val QWK: 0.9088, Val Acc: 84.45%                   |
| Visualization generation          | ✅ Complete | Training curves, confusion matrix, domain analysis |
| Training stability verification   | ✅ Complete | AMP-enabled, stable convergence, no NaN losses     |

**Phase 1 Results Summary**:
- **Best validation QWK**: 0.9088 (epoch 14) — significantly exceeds expected range (0.75-0.82)
- **Best validation accuracy**: 84.45% (epoch 14)
- **Best validation AUC**: 0.9846 (epoch 14)
- **Training set**: 2,929 images | **Validation set**: 733 images
- **Artifacts location**: `dr-results/outputs/` (separate from CPU runs in `outputs/`)
- **Key observation**: Baseline ResNet-50 achieves strong performance, setting a high bar for CBAM ablation study

### 11.3. Pending Work

| Task                                           | Status             | Priority     | Dependencies                |
| ---------------------------------------------- | ------------------ | ------------ | --------------------------- |
| Messidor-2 evaluation (baseline checkpoint)    | ⬜ Not started     | **Blocking** | Phase 1 complete ✅          |
| GPU CBAM-ResNet50 training (fold 0)            | ⬜ Not started     | **Blocking** | Phase 1 complete ✅          |
| GPU CBAM-ResNet50 training (folds 1-4)         | ⬜ Not started     | **Blocking** | Fold 0 CBAM complete        |
| Cross-fold statistics (mean ± std)             | ⬜ Not started     | **Blocking** | All 5 folds complete        |
| ECE / calibration metrics in evaluate.py       | ⬜ Not implemented | High         | —                           |
| Referral threshold analysis                    | ⬜ Not implemented | High         | Messidor-2 evaluation       |
| Ablation study (4 model variants)              | ⬜ Not started     | High         | Baseline + CBAM fold 0      |
| Messidor-2 evaluation (CBAM checkpoint)        | ⬜ Not started     | High         | CBAM training complete      |
| Uncertainty visualizations                     | ⬜ Not generated   | Medium       | Messidor-2 evaluation       |
| Publication-quality figure refinement          | ⬜ Not generated   | Medium       | All experiments complete    |
| Thesis document writing                        | ⬜ Not started     | High         | Core experiments complete   |

**Next Immediate Steps** (Phase 2):
1. Run Messidor-2 external validation on baseline checkpoint (`dr-results/outputs/checkpoints/baseline_resnet50_fold0_best.pth`)
2. Train CBAM-ResNet50 on fold 0 with identical hyperparameters
3. Compare baseline vs. CBAM performance on both APTOS validation and Messidor-2

---

## 12. Identified Issues & Fixes

### Issue 1 — Validation Uses MC Dropout (Stochastic Noise)

**Severity**: MEDIUM | **Status**: ✅ FIXED

**Problem**: `MCDropout.forward()` hard-coded `training=True`, making validation during training non-deterministic. Best model selection could be based on a lucky dropout mask.

**Fix**: Added `mc_active` toggle and `deterministic_mode()` context manager. Validation now uses deterministic mode; MC inference retains stochasticity.

### Issue 2 — Missing Gradient Clipping

**Severity**: LOW | **Status**: ✅ FIXED

**Problem**: Focal Loss with γ=2.0 amplifies gradients on hard examples, some of which are mislabelled in APTOS.

**Fix**: Added `scaler.unscale_(optimizer)` + `clip_grad_norm_(max_norm=1.0)` in training loop.

### Issue 3 — Progress Bar Epoch Display

**Severity**: LOW | **Status**: ✅ FIXED

**Problem**: Progress bar hard-coded `EPOCHS` constant instead of actual `--epochs` argument.

**Fix**: `self.num_epochs` stored in Trainer, updated in `fit()`.

### Issue 4 — Baseline Used Different Hyperparameters

**Severity**: MEDIUM | **Status**: ✅ FIXED

**Problem**: Original baseline (from notebook) trained with batch_size=4, CPU, 5 epochs — confounds ablation comparison.

**Fix**: Created `BaselineResNet50` with identical head structure. Both models use same `train.py` pipeline. Only difference: CBAM on vs. off.

### Issue 5 — ECE Not Computed in evaluate.py

**Severity**: HIGH (for thesis quality) | **Status**: ⬜ NOT FIXED

**Problem**: `evaluate.py` computes accuracy, QWK, AUC but not ECE or Brier score. Since the thesis's core claim involves uncertainty, omitting calibration metrics is a critical gap.

**Fix needed**: Add `compute_ece()` function and reliability diagram plot.

### Issue 6 — GaussNoise API Deprecation

**Severity**: LOW | **Status**: ⬜ NOT FIXED

**Problem**: `A.GaussNoise(var_limit=...)` API changed in Albumentations ≥ 1.4.

**Fix**: Pin version in `requirements.txt` or update to `A.GaussNoise(std_range=...)`.

### Issue 7 — No Per-Patient Split

**Severity**: MEDIUM (for thesis defensibility) | **Status**: ⚠️ UNFIXABLE

**Problem**: APTOS 2019 does not provide patient IDs. Per-image splits may inflate metrics (Voets 2019).

**Mitigation**: Document as known limitation. Messidor-2 used as unseen external test (no split needed).

### Issue 8 — Messidor-2 Label Provenance

**Severity**: LOW (standard practice) | **Status**: ✅ RESOLVED

**Problem**: The official Messidor-2 dataset was released without DR severity grades. Third-party labels are required for evaluation. The project originally contained `trainLabels.csv` from the Kaggle EyePACS competition (35,127 entries) — an entirely different dataset — misidentified as Messidor-2 labels.

**Fix**: Replaced with adjudicated labels from Krause et al. (2018), sourced from [Kaggle google-brain/messidor2-dr-grades](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades). Dataset code updated to read `image_id` / `adjudicated_dr_grade` columns and filter ungradable images. The "ground truth" for external validation is derived from expert consensus rather than definitive clinical diagnosis — this is a common limitation across all Messidor-2 evaluation studies.

---

## 13. Literature Gaps & Novel Contributions

### Gap 1 — Calibration Reporting (ECE/Brier)

**Impact**: HIGH | **Effort**: LOW

The Chopra 2025 survey explicitly calls out the absence of calibration metrics in DR papers. Simply adding ECE computation to `evaluate.py` closes this gap. Implementation: bin-based ECE + reliability diagram.

### Gap 2 — Uncertainty-Aware Referral Thresholding

**Impact**: HIGH | **Effort**: MEDIUM

No DR paper derives a specific entropy threshold for "refer to human grader" decisions and reports sensitivity/specificity at that threshold. Implementation: compute accuracy/coverage trade-off at different entropy percentiles, identify optimal threshold (e.g., best accuracy at ≥85% coverage).

### Gap 3 — Cross-Dataset CBAM Ablation

**Impact**: MEDIUM | **Effort**: LOW (already designed)

No paper ablates CBAM on/off across APTOS → Messidor-2. The project design directly addresses this with baseline ResNet-50 vs. CBAM-ResNet50 evaluated on both datasets.

### Gap 4 — Ordinal Loss Functions (Potential Future Work)

**Impact**: HIGH | **Effort**: MEDIUM

No reviewed paper applies ordinal regression losses (ordinal CE, CORN) to CBAM-CNN on APTOS/Messidor-2. Standard Focal Loss treats DR grades as independent classes, ignoring that grade 3 is closer to grade 4 than to grade 0.

---

## 14. Failure Modes & Risk Assessment

### 14.1. Technical Failure Modes

| #   | Failure Mode                                          | Risk       | Evidence                       | Mitigation                                                      |
| --- | ----------------------------------------------------- | ---------- | ------------------------------ | --------------------------------------------------------------- |
| FM1 | Focal Loss × APTOS label noise interaction            | MEDIUM     | Chopra 2025 Section IX         | Monitor per-class loss curves; consider label smoothing (ε=0.1) |
| FM2 | Weak Bayesian approximation (single-layer MC Dropout) | HIGH       | Inference (no T1 evidence)     | Report ECE honestly; propose backbone-level dropout as fix      |
| FM3 | CBAM spatial attention too coarse at deep layers      | LOW-MEDIUM | Inference (IDRiD lesion sizes) | Acknowledge in Discussion; multi-scale CBAM mitigates partially |
| FM4 | Alpha weights computed per-fold vs. globally          | LOW        | Code inspection                | ✅ Already correct — alpha computed from training fold          |

**FM2 Detail**: MC Dropout at p=0.5 on the final Linear(2048→5) layer only masks 1,024/2,048 features per pass, but all features are identically extracted by the deterministic backbone. This produces narrow uncertainty estimates that may be poorly calibrated.

### 14.2. Project Risk Assessment

| Risk                                        | Probability | Impact       | Mitigation                                                     | Status          |
| ------------------------------------------- | ----------- | ------------ | -------------------------------------------------------------- | --------------- |
| GPU not available for full training         | ~~Medium~~  | **Blocking** | ~~Vast.ai ($5–7 total), or reduce to 1–2 folds~~               | ✅ **Resolved** |
| CBAM shows no improvement over baseline     | Medium      | High         | Frame as negative result with ablation — still publishable     | Active risk     |
| Messidor-2 performance very low (QWK < 0.5) | Medium      | Medium       | Expected due to domain shift — discussion point, not failure   | Active risk     |
| MC Dropout entropy poorly calibrated        | High        | Medium       | Report ECE honestly; propose backbone-level dropout            | Active risk     |
| Time runs out before all 5 folds            | High        | Low          | 1 fold + external validation sufficient for bachelor's         | Mitigated       |
| Focal Loss overfits to label noise          | Low         | Medium       | Monitor per-class loss; label smoothing ablation               | Monitored       |
| Baseline performance too strong             | **New**     | Medium       | May reduce CBAM's relative improvement; still valid comparison | Emerging risk   |

**Risk Update (2026-03-31)**: Baseline ResNet-50 achieved QWK 0.9088, significantly exceeding expected range (0.75-0.82). This raises the bar for CBAM-ResNet50 to demonstrate improvement. If CBAM shows minimal gains, the thesis can reframe as "validating strong baseline performance" rather than "improving via attention mechanisms."

---

## 15. Phased Implementation Plan

### Phase 0: Infrastructure Fixes — ✅ COMPLETE

Fixed MCDropout deterministic validation, gradient clipping, progress bar, baseline model factory, CLI arguments, experiment config system. All CPU smoke tests passed.

### Phase 1: Baseline Retraining — ✅ COMPLETE (2026-03-31)

**Completed**:
- ✅ Trained baseline ResNet-50 (fold 0, GPU, 19 epochs, batch_size=16)
  - GPU: NVIDIA RTX 3090 (25.3GB VRAM) via Vast.ai
  - Training time: ~2.5 hours
  - Best validation QWK: **0.9088** (epoch 14) — exceeds expected range
  - Best validation accuracy: **84.45%** (epoch 14)
  - Best validation AUC: **98.46%** (epoch 14)
- ✅ Generated training artifacts (checkpoints: 270MB × 2, logs, metrics)
- ✅ Generated 19 visualization figures (training curves, confusion matrix, domain analysis)
- ✅ Results location: `dr-results/outputs/` (separate from CPU runs)

**Pending**:
- ⬜ Evaluate baseline on Messidor-2 (MC Dropout T=20) — next priority
- ⬜ Record external validation metrics: QWK, AUC, accuracy, entropy statistics

### Phase 2: CBAM-ResNet50 Training (Days 3–5)

- Train CBAM-ResNet50 fold 0, compare with baseline
- Train folds 1–4, compute cross-fold statistics (mean ± std)

### Phase 3: Evaluation & Uncertainty Analysis (Days 6–8)

- Add ECE to evaluate.py + reliability diagram
- Messidor-2 evaluation (both models)
- Referral threshold analysis
- Uncertainty visualizations and case studies

### Phase 4: Ablation Study (Days 9–10)

- Train 4 model variants (A–D) on fold 0
- Evaluate all on APTOS val + Messidor-2
- Fill ablation table

### Phase 5: Visualization & Figures (Days 11–12)

- Training curves, confusion matrices, ROC curves
- Uncertainty figures, architecture diagram

### Phase 6: Additional Experiments (Days 13–14, if time permits)

- Ordinal loss experiment, backbone-level MC Dropout, Grad-CAM

### Phase 7: Thesis Writing (Days 15–20)

- Chapters 1–6, references, appendix

### Phase 8: Final Polish (Days 21–22)

- Code cleanup, README, thesis review, submission

---

## 16. Expected Results

### 16.1. APTOS Validation

| Metric       | Baseline ResNet-50 | CBAM-ResNet50 (Expected) |
| ------------ | ------------------ | ------------------------ |
| Val Accuracy | 72–78%             | 78–85%                   |
| Val QWK      | 0.75–0.82          | 0.75–0.85                |
| Macro F1     | 0.55–0.65          | 0.60–0.72                |
| Val AUC      | 0.85–0.92          | 0.88–0.94                |

### 16.2. Messidor-2 External

| Metric               | Baseline  | CBAM-ResNet50 (Expected) |
| -------------------- | --------- | ------------------------ |
| Accuracy             | 65–75%    | 68–78%                   |
| QWK                  | 0.55–0.70 | 0.60–0.75                |
| Binary Referable AUC | 0.85–0.90 | 0.88–0.93                |

> **Note**: 5–15% drop from APTOS is expected due to domain shift and is a key discussion point.

### 16.3. Uncertainty Metrics (Expected Patterns)

- Incorrectly predicted images should have higher entropy than correct predictions.
- Boundary classes (Mild, grade 1) should show higher entropy than No DR (grade 0).
- Inverse correlation between confidence and entropy.

---

## 17. Thesis Structure

| Chapter   | Content                                                | Estimated Pages  |
| --------- | ------------------------------------------------------ | ---------------- |
| 1         | Introduction (background, objectives, scope)           | 5–7              |
| 2         | Literature Review + Theoretical Foundation             | 15–20            |
| 3         | Methodology (architecture, data, training, evaluation) | 12–15            |
| 4         | Experimental Results                                   | 10–15            |
| 5         | Discussion (analysis, limitations, comparison)         | 5–8              |
| 6         | Conclusion & Future Work                               | 3–5              |
| —         | References (~30+ papers)                               | 3–5              |
| —         | Appendix (code, additional figures)                    | 5–10             |
| **Total** |                                                        | **~60–80 pages** |

### Key Limitations to Discuss

1. Per-image splits (APTOS lacks patient IDs) — may inflate metrics.
2. Single-layer MC Dropout may underestimate uncertainty.
3. CBAM spatial attention too coarse at deep layers for individual MA localisation.
4. APTOS label noise (single grader) interacts with Focal Loss.
5. Small dataset (~3.6K images) limits generalization claims.

---

## 18. Future Work

1. **Ordinal loss functions** (CORN / ordinal CE) — respects grade ordering, may improve QWK.
2. **Backbone-level MC Dropout** (p=0.1 before avg-pool) — broader uncertainty estimates.
3. **Grad-CAM / attention map visualization** — interpretability and clinical trust.
4. **Multi-task learning** — simultaneous grading + lesion segmentation (addresses Alyoubi 6% gap).
5. **Foundation model fine-tuning** (RETFound) — label-efficient, stronger baseline.
6. **Progression prediction** (DeepDR Plus approach) — time-to-event, personalized screening intervals.
7. **Federated learning** — privacy-preserving multi-centre training.
8. **Label smoothing** — mitigate APTOS label noise.
9. **Temperature scaling** — post-hoc calibration improvement.

---

## 19. References

### Core Architecture & Methods

1. He, K. et al. "Deep Residual Learning for Image Recognition." CVPR, 2015.
2. Woo, S. et al. "CBAM: Convolutional Block Attention Module." ECCV, 2018.
3. Gal, Y. & Ghahramani, Z. "Dropout as a Bayesian Approximation." ICML, 2016.
4. Lin, T.-Y. et al. "Focal Loss for Dense Object Detection." ICCV, 2017.
5. Graham, B. "Kaggle Diabetic Retinopathy Detection Competition, 1st Place." 2015.

### DR-Specific Studies

6. Gulshan, V. et al. "Development and Validation of a Deep Learning Algorithm for Detection of DR." JAMA, 2016.
7. Voets, M. et al. "Reproduction Study Using Public Data of: Development and Validation..." arXiv, 2019.
8. Teng et al. "MediDRNet." 2024. (CBAM + prototypical contrastive for DR)
9. Bhati et al. "IDANet: Interpretable Dual Attention Network." Artificial Intelligence in Medicine, 2024.
10. Romero-Oraá et al. "Attention-Based Separate Dark/Bright Structure Attention." 2024.
11. Akram et al. "Bayesian Approaches for DR Grading." 2025.

### Surveys & Reviews

12. Chopra, M. et al. "From Retinal Pixels to Patients: Evolution of Deep Learning Research in DR Screening." arXiv:2511.11065, 2025.
13. Alyoubi et al. "DR Detection Through Deep Learning Techniques: A Review." Informatics in Medicine Unlocked, 2020.
14. Aburass, S. et al. "Vision Transformers in Medical Imaging: Comprehensive Review." J Imaging Inform Med, 2025.
15. "Systematic Review of Regulator-Approved Deep Learning Systems for DR Screening." Nature Digital Medicine, 2025.
16. Bappi et al. "Deep Learning-Based DR Recognition: Survey." Healthcare Analytics, 2025.
17. "Systematic Review of Hybrid Vision Transformer Architectures." J Digit Imaging, 2025.

### Foundation Models & Advanced Methods

18. Zhou et al. "RETFound: A Foundation Model for Retinal Imaging." Nature, 2023.
19. Dai et al. "DeepDR Plus: Predicting Time to DR Progression." Nature Medicine, 2024.
20. "DRAMA: Multi-Label Learning for DR." Acta Ophthalmologica, 2025.
21. "Multi-Task Learning for DR Grading and Lesion Segmentation." AAAI, 2023.

### Datasets

22. APTOS 2019 Blindness Detection. Kaggle Competition, 2019.
23. Decencière, E. et al. "Feedback on a Publicly Distributed Image Database: the Messidor Database." Image Analysis & Stereology, 2014.
24. Krause, J. et al. "Grader Variability and the Importance of Reference Standards for Evaluating Machine Learning Models for Diabetic Retinopathy." Ophthalmology, vol. 125, no. 8, 2018, pp. 1264–1272.
25. DDR Dataset. 2019.
26. IDRiD Dataset. 2018.

### Generative & Domain Adaptation

26. Kim et al. "StyleGAN for Retinal Synthesis." Nature Scientific Reports, 2022.
27. "CCDM: Conditional Cascaded Diffusion Model." Nature Communications Medicine, 2025.
28. Xia et al. "DECO: Domain Disentanglement." MICCAI, 2024.

### Additional

29. Bhoopalan et al. "TOViT: Task-Optimized Vision Transformer." Nature Scientific Reports, 2025.
30. Huang et al. "Lesion-Aware Contrastive Pretraining." MICCAI, 2021.
31. Yang et al. "MAE for ViT Pretraining on DR." PLOS ONE, 2024.
32. Matta et al. "Federated Learning for DR." Scientific Reports, 2023.

---

_Document consolidates information from: `ideas/evaluate.md`, `ideas/project-overview.md`, `ideas/research.md`, `plans/02-phase1-3.md`, `docs/context-paper/dr-reasearch-context.md`, `docs/context-paper/essential-papers-dr-detection.md`, `docs/VASTAI_DEPLOYMENT.md`, and all source code in `src/`._
