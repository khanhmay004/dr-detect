## Paper-by-paper intake (dr-research protocol)

---

### Gulshan et al. — JAMA, 2016

**Problem framing** Landmark proof-of-concept: can a CNN trained on a very large private fundus corpus detect referable DR (grades ≥ 2, binary task) at near-ophthalmologist performance? Screening framing — sensitivity-optimised.

**Methodology**

- Input: colour fundus, large private corpus (~128K), BEN normalisation implied
- Architecture: Inception-v3, fine-tuned from ImageNet
- Loss: binary cross-entropy on adjudicated labels (multi-grader)
- No domain-specific design choices beyond scale

**Key results**

| Dataset    | Metric | This paper |
| ---------- | ------ | ---------- |
| EyePACS    | AUC    | 0.991      |
| Messidor-2 | AUC    | 0.990      |

**Limitations** Private dataset not released. Voets et al. (2019) could only reproduce AUC 0.85–0.95 on public labels. Per-image not per-patient splits. `[Reviewer note]` Sensitivity 0.90 at specificity 0.98 — operating point not standardised.

**Reproducibility** Code: Not released. Dataset: Private.

---

### Voets et al. — arXiv, 2019

**Problem framing** Reproduction study of Gulshan et al. using only publicly available data and labels. Directly surfaces the reproducibility crisis.

**Key results**

| Dataset    | Metric | Gulshan claimed | Voets reproduced |
| ---------- | ------ | --------------- | ---------------- |
| EyePACS    | AUC    | 0.99            | 0.95             |
| Messidor-2 | AUC    | 0.99            | 0.85             |

**Critical finding** Three factors explain the gap: (1) private curation steps, (2) per-image rather than per-patient splits, (3) label adjudication differences. This paper is **essential reading** for your methodology chapter — it is the canonical argument for why you must use per-patient splits.

---

### From Retinal Pixels to Patients — Chopra et al., arXiv:2511.11065, November 2025

**Problem framing** First systematic survey spanning 2016–2025: 50+ studies, 20+ datasets, consolidating reproducibility, domain shift, class imbalance, architecture evolution (CNN→ViT→neuro-symbolic), federated learning, and benchmarking into one taxonomy.

**Methodology** Not a method paper — systematic review with narrative synthesis and two consolidated tables (Table I: datasets; Table II: performance across methods).

**Key contributions**

- First survey to link reproducibility failures, domain shift, and clinical deployment barriers under one framework
- Evaluation protocol recommendations: per-patient splits, external validation, calibration metrics (ECE, Brier)
- Research agenda for 2025+: open multi-center benchmarks, lightweight models, federated privacy-preserving deployment

**Critical benchmark table (from paper, Table II, selected)**

| Paper        | Dataset    | Task    | AUC  | Acc  | Notes                 |
| ------------ | ---------- | ------- | ---- | ---- | --------------------- |
| Gulshan 2016 | EyePACS    | Binary  | 0.99 | —    | Private data          |
| Voets 2019   | Messidor-2 | Binary  | 0.85 | —    | Reproducibility study |
| Arrieta 2023 | EyePACS    | Binary  | 0.94 | —    | SSL, 2% labels        |
| Yang 2024    | APTOS      | Binary  | 0.98 | 0.93 | MAE ViT               |
| Bhati 2024   | DDR        | 5-class | 0.86 | 0.85 | IDANet attention      |
| Zhou 2025    | APTOS+Mess | 5-class | 0.99 | 0.94 | GPMKLE-Net            |

**Limitations** Survey scope ends at public data — privately deployed systems (FDA-cleared) are excluded. No meta-analysis or pooled statistics.

---

### Vision Transformers in Medical Imaging — Aburass et al., J Imaging Inform Med, 2025

**Problem framing** Comprehensive review of ViT variants applied to medical imaging tasks including DR classification. Mathematical formulations of patch embedding, self-attention, positional encoding.

**DR-specific results (Section from paper)**

| Study               | Architecture                  | Dataset       | Accuracy |
| ------------------- | ----------------------------- | ------------- | -------- |
| Wu et al. 2021      | Vanilla ViT                   | Not specified | 91.4%    |
| Adak et al. 2023    | Ensemble (ViT+BEiT+CaiT+DeiT) | APTOS         | High     |
| Oulhadj et al. 2024 | ViT + CapsuleNet              | APTOS         | 88.18%   |
| Oulhadj et al. 2024 | ViT + CapsuleNet              | EyePACS       | 78.64%   |

**Reviewer note** The EyePACS performance (78.64%) for the ViT+Capsule hybrid is notably lower than APTOS — consistent with the general pattern of APTOS overfitting. This paper should be cited in your related work when introducing ViT architectures, but be careful: it is a review paper, not primary research, so cite the underlying studies (Wu, Adak, Oulhadj) for specific numbers.

---

### RETFound — Zhou et al., Nature, 2023

**Problem framing** First foundation model for retinal imaging. Self-supervised masked autoencoder (MAE) pre-trained on 904,170 colour fundus photographs + 736,442 OCT images from Moorfields Eye Hospital. Label-efficient fine-tuning for downstream tasks including DR grading.

**Methodology**

- Input: colour fundus + OCT, mixed modality pre-training
- Architecture: ViT-L/16, masked autoencoder (75% masking ratio)
- Training objective: reconstruction of masked patches — self-supervised
- Fine-tuning: small labelled subsets (as few as 10% of training data)

**Key results**

| Dataset    | Task            | AUC   |
| ---------- | --------------- | ----- |
| APTOS 2019 | 5-class grading | 0.943 |
| IDRiD      | 5-class grading | 0.822 |
| Messidor-2 | Referable DR    | 0.884 |

**Limitations** ViT-L/16 is large (~307M parameters) — inference cost and VRAM requirements are high for clinical deployment. Pre-training data is from a single institution (Moorfields, UK) — potential demographic bias. Authors acknowledge performance on Messidor-2 (0.884) is lower than on APTOS (0.943), likely due to domain shift from the Moorfields pre-training distribution.

**Reproducibility** Code: https://github.com/rmaphoh/RETFound. Pre-trained weights: released publicly. This is the strongest open-source baseline for your thesis comparison.

---

### Systematic Review of FDA-Approved DR Systems — Nature Digital Medicine, December 2025

**Problem framing** Meta-analysis of 82 studies covering 887,244 examinations across 25 devices and 28 countries — the largest real-world validation study of AI-based DR screening systems.

**Key aggregate results**

- Sensitivity: 93% (pooled)
- Specificity: 90% (pooled)
- Factors improving performance: pupil dilation, adjudicated reference standards, portable cameras

**Critical finding for your thesis** This paper establishes the clinical deployment bar: 93% sensitivity / 90% specificity at the referable threshold. Any model you evaluate should be benchmarked against this. It also confirms that 25+ FDA-cleared devices exist — the field has moved from research to deployment, and your thesis sits in this transition.

**Reviewer note** The paper covers only regulatory-cleared systems — open-source research models (including your thesis model) are not included. This is an important contextual caveat.

---

### Deep Learning–Based DR Recognition Survey — Bappi et al., Healthcare Analytics (Elsevier), 2025

**Problem framing** Survey focused on attention mechanisms and feature fusion strategies for DR classification (2020–2024).

**Key contributions** Catalogs attention-based architectures, multi-task learning approaches, feature fusion techniques. Identifies challenges: class imbalance, cross-domain generalization, interpretability.

**Reviewer note** Less comprehensive than the Chopra et al. (2025) survey on DR specifically. Useful for attention mechanism citations but not as a primary reference.

---

### Intelligent Retinal Disease Detection — Nature Scientific Reports, December 2025

**Problem framing** Hybrid ensemble: MobileNetV2 + DenseNet121 with feature fusion using PCA (450 features) + DWT (12 features per image). Multi-disease classification on BRSET dataset (16,266 images, 8,524 patients).

**Key results**

- Accuracy: 98.2% on multi-disease classification
- Dataset: BRSET (NOT APTOS or Messidor-2 — limits comparability)

**Reviewer note** The 98.2% figure is on BRSET, a relatively clean multi-class dataset. No external validation reported. PCA+DWT feature fusion is an unusual engineering choice — no principled argument is given for why DWT features complement deep features.

---

### Swin-L vs CNN Comparison — Bioengineering (PMC12383659), August 2025

**Problem framing** Head-to-head comparison of 6 architectures (3 CNNs: ResNet, EfficientNet, DenseNet; 3 ViTs: Swin-L, Twins-SVT-L, CSWin-B) on BRSET dataset for binary and 3-class DR grading.

**Key results**

| Model             | Params | Binary DR AUC | 3-Class DR AUC | PDR AUC |
| ----------------- | ------ | ------------- | -------------- | ------- |
| Swin-L            | 197M   | 0.98          | 0.98           | 0.99    |
| Twins-SVT-L       | 99M    | 0.98          | 0.97           | —       |
| CSWin-B           | 78M    | 0.97          | 0.97           | —       |
| ResNet (baseline) | ~25M   | Lower         | Lower          | —       |

**Critical note for your thesis** Swin-L's 197M parameters vs. ResNet-50's ~24M is an 8× parameter overhead. On APTOS 2019 (3,662 images), Swin-L would almost certainly overfit without careful regularization or pre-training. CSWin-B at 78M is a more reasonable comparison point. The BRSET dataset (16,266 images) is considerably larger than APTOS — results may not transfer.

---

### Systematic Review of Hybrid ViT Architectures — J Digit Imaging (PMC12572492), January 2025

**Problem framing** PRISMA-guided review of 34 hybrid CNN-Transformer architectures across medical imaging tasks. Categorizes merging strategies: feature concatenation, attention fusion, adaptive gating.

**Key finding** Hybrid models consistently outperform pure CNNs and pure ViTs on small-to-medium medical datasets. CNN local features + ViT global context = complementary strengths.

**Reviewer note** This is directly relevant to your thesis if you add a ViT comparison. The review supports the design decision to use CBAM attention in a CNN backbone rather than switching wholesale to ViT — `[Inference]` a well-tuned CBAM-ResNet50 captures some of the local+global benefit at far lower parameter cost than a full hybrid model.

---

### Alyoubi et al. Review — Informatics in Medicine Unlocked, 2020

**Problem framing** Review of 33 papers (2016–2019). Identifies the critical 6% gap: only 6% of reviewed papers perform both severity grading AND lesion detection simultaneously.

**Key contributions** Classification of approaches: binary, multi-level, lesion-based, vessel-based. Dataset descriptions for Kaggle, Messidor, DDR, IDRiD, APTOS, E-ophtha. Preprocessing survey: CLAHE, green channel, augmentation.

**Reviewer note** This is the foundational historical review for your thesis. Cite it in the introduction for pre-2020 context, and to establish the 6% lesion detection gap that motivates multi-task learning approaches.

---

### IDANet — Bhati et al., Artificial Intelligence in Medicine, 2024

**Problem framing** Interpretable Dual Attention Network for DR grading. Embeds channel and spatial attention directly into the training pipeline rather than using post-hoc saliency.

**Methodology**

- Backbone: CNN (not specified precisely in survey)
- Attention: dual attention (channel + spatial, analogous to CBAM but integrated differently)
- Dataset: DDR (5-class), EyePACS (5-class), IDRiD

**Key results**

| Dataset | Task    | AUC  | Accuracy |
| ------- | ------- | ---- | -------- |
| DDR     | 5-class | 0.86 | 0.85     |
| EyePACS | 5-class | 0.96 | 0.93     |
| IDRiD   | 5-class | 0.93 | 0.90     |

**Critical evaluation for your thesis** IDANet is the most direct architectural comparator to your CBAM-ResNet50. Both use dual attention (channel + spatial). Key difference: your project places CBAM after each ResNet stage; IDANet's exact placement is not specified in the survey description. If you can access the IDANet paper, comparing CBAM placement strategies would strengthen your methodology section.

---

### MediDRNet — Teng et al., 2024

**Problem framing** Addresses severe class imbalance specifically for minority classes (Severe NPDR, PDR). Dual-branch network: feature learning + classifier branches. Prototypical contrastive learning — minimise distance to class prototype, maximise distance from other class prototypes. CBAM for subtle lesion features.

**Key results** SOTA on Kaggle dataset for minority category performance (Severe NPDR, PDR). Strong on ultra-wide field (UWF) dataset.

**Critical evaluation for your thesis** MediDRNet uses CBAM — directly comparable to your architecture. However, it adds prototypical contrastive learning on top of CBAM, which is a training strategy your project does not use. This creates a clean ablation opportunity: your CBAM-ResNet50 + Focal Loss vs. MediDRNet's CBAM + Focal Loss + prototypical contrastive loss. If you can run this ablation, it would be a genuine contribution.

---

### Attention-Based Separate Dark/Bright Structure Attention — Romero-Oraá et al., 2024

**Problem framing** Explicitly separates attention for dark lesions (microaneurysms, haemorrhages) and bright lesions (hard exudates). Image decomposition before attention. Xception backbone + focal loss.

**Key results**

- Accuracy: 83.7% on Kaggle
- QWK: 0.78

**Critical evaluation** The clinical motivation is strong: dark and bright lesions respond differently to preprocessing and carry different clinical significance. The separation is semantically grounded, not just a trick. Your CBAM applies single attention maps — it does not separate by lesion type. This is a genuine gap in your architecture that you should acknowledge in the Discussion section.

---

### Multi-Task Learning for DR Grading + Lesion Segmentation — AAAI 2023

**Problem framing** Addresses the Alyoubi 6% gap directly — simultaneous DR severity grading AND lesion segmentation (microaneurysms, haemorrhages, exudates). Semi-supervised learning to obtain segmentation masks. Outperforms single-task SOTA.

**Key results** Outperforms single-task networks on both grading (QWK) and segmentation (Dice) tasks.

**Critical evaluation for your thesis** Your project explicitly excludes lesion segmentation (stated in project-overview.md Section 8.4). This is a legitimate scope decision for a 1-month thesis. However, you should cite this paper in the Discussion as the direction that would extend your work — it directly addresses the key gap your approach leaves open.

---

### DRAMA — Acta Ophthalmologica, 2025 (PMC12167062)

**Problem framing** 11-task multi-label framework: image quality, lesion detection (MA, haemorrhages, exudates, cotton wool spots), DR severity, diabetic macular edema. EfficientNet-B2 backbone.

**Key results**

- Quality assessment: 87.02%
- Lesion detection: 91.60%
- AUC >0.95 for most tasks
- Speed: 86ms for full test set (vs 90–100 min for human graders)

**Critical evaluation** DRAMA demonstrates that multi-task learning is feasible at scale and is clinically faster by orders of magnitude. EfficientNet-B2 is a stronger backbone than ResNet-50 for this scale. `[Reviewer note]` The 86ms speed figure is for the full test set, not per-image — this is likely the batch inference time, which overstates clinical throughput. Per-image latency would need to be reported for deployment comparison.

---

### MVCAViT — Nature Scientific Reports, 2025 (DOI: s41598-025-18742-z)

**Problem framing** Dual-view learning: macula-centered + optic-disc-centered fundus images. Cross-attention mechanism between views. Multi-task: disease classification + lesion localization + severity grading. Particle Swarm Optimization (PSO) for hyperparameter tuning.

**Critical evaluation** The dual-view approach is architecturally interesting but requires two images per patient. Standard DR screening datasets (APTOS, Messidor-2) typically provide single-view images. `[Inference]` MVCAViT would not be directly replicable on APTOS without additional preprocessing to simulate dual views.

---

### DeepDR Plus — Dai et al., Nature Medicine, 2024

**Problem framing** Progression prediction — time-to-event for DR worsening within 5 years. Pre-trained on 717,308 images. Extends screening interval from 12 months to 31.97 months for low-risk patients.

**Key results**

- Concordance Index: 0.754–0.846
- Delayed detection rate: only 0.18% for vision-threatening DR

**Critical evaluation for your thesis** DeepDR Plus is explicitly listed in your project as a Future Work direction. The concordance index metric and time-to-event framing are fundamentally different from the 5-class grading task in your thesis. Cite it in Future Work, but do not conflate its metrics with your grading metrics.

---

### StyleGAN for Retinal Synthesis — Kim et al., Nature Scientific Reports, 2022

**Problem framing** High-quality synthetic retinal image generation via StyleGAN. Turing test: ophthalmologists 54% accuracy (near-random). AUC improved 23.7% (0.735→0.909) for extremely imbalanced class (1:0.1 ratio).

**Critical evaluation** The 23.7% AUC improvement is for an extreme 1:0.1 imbalance, which is more severe than APTOS 2019 (worst class is ~5%, not 1.0%). The survey (Chopra 2025) notes that GAN-based gains often "reflected artifacts rather than pathology." Focal Loss (which your project uses) is a simpler and more principled approach to class imbalance without this risk.

---

### CCDM — Nature Communications Medicine, 2025

**Problem framing** Conditional cascaded diffusion model generating synthetic retinal images conditioned on patient demographics and DR stage. FID = 9.3.

**Key results**

- Internal test: AUROC improved 0.827 → 0.851 (+0.024) with ×2 synthetic augmentation
- External test: no significant improvement

**Critical evaluation for your thesis** The external test failure is the critical result: CCDM improves internal performance but fails to generalize. This is consistent with Chopra et al.'s warning about GAN/diffusion augmentation embedding site-specific artifacts. For your thesis: Focal Loss + stratified sampling is safer than synthetic augmentation. If you do test synthetic augmentation, validate externally on Messidor-2.

---

Now the cross-paper comparison and methodology taxonomy:---

## Methodology taxonomy (dr-research Section 4b)

Following the skill's five axes:

**Architecture family.** The literature has moved through three clear phases. 2016–2021 is dominated by CNN backbones (Inception-v3, ResNet-50, EfficientNet variants) with ImageNet pre-training and simple classification heads. 2022–2024 introduces hybrid models (CNN + attention modules like CBAM, dual attention) and early ViT applications to DR. 2025 onward sees full ViT deployment (Swin-L, TOViT), neuro-symbolic hybrids (Urooj 2025), and foundation models (RETFound). Your CBAM-ResNet50 sits firmly in the 2022–2024 hybrid tier — well-supported by the literature, not cutting-edge for 2025.

**Training objective.** Standard cross-entropy remains most common. Focal loss appears in a meaningful subset (Lin 2017, Romero-Oraá 2024, your thesis). Ordinal regression losses (which respect the 0→4 grade ordering) appear in very few DR papers — this is a genuine gap (see Section below). Self-supervised and masked autoencoder objectives are present in RETFound and Yang 2024. Contrastive learning (Huang 2021, Alam 2023, MediDRNet 2024) is a growing direction.

**Label handling.** Most papers assume clean labels. Noisy label learning is rare in the DR space despite APTOS 2019's known noise. Semi-supervised learning appears in a few papers (Duan 2022, Arrieta 2023). Ensemble adjudication is mentioned but rarely implemented in open-source work.

**Data regime.** Fully supervised on APTOS/EyePACS dominates. Label-efficient SSL is a meaningful minority (Huang 2021, Arrieta 2023, Yang 2024, RETFound). Few-shot learning appears as a single paper (balanced episodic learning 2025). No published work on one-shot DR grading.

**Preprocessing.** BEN (Ben Graham) normalisation and CLAHE are the two dominant approaches, occasionally combined. Your thesis uses BEN — this is well-supported. Green channel extraction appears in older work. ESRGAN super-resolution has been used (Alwakid 2023) but the survey warns its gains may reflect artifacts.

---

## Literature gap analysis (dr-research Section 4c)

**Gap 1 — Ordinal loss functions on APTOS/Messidor-2.** No reviewed paper applies ordinal regression losses (ordinal CE, Cumulative Link Models, CORN) to a CNN backbone with CBAM attention on these datasets. IDANet and MediDRNet both use standard CE + focal loss despite the ordinal grade structure. Standard focal loss treats DR grades as independent classes and ignores that grade 3 is much closer to grade 4 than to grade 0. Ordinal losses have improved QWK in analogous skin lesion grading tasks (T2 evidence). This is a clean gap with low implementation overhead — swapping the loss function in your existing training loop would produce a publishable ablation.

Proposed research question: _Does replacing focal loss with an ordinal-aware loss improve quadratic weighted kappa for 5-class DR grading on APTOS 2019 and Messidor-2, controlling for backbone and preprocessing?_

**Gap 2 — Calibration reporting for DR grading models.** The Chopra 2025 survey explicitly calls out the absence of calibration metrics (ECE, Brier score) in most DR papers. Models may be miscalibrated even when AUC is high — a model that reports 95% confidence but is correct only 70% of the time is dangerous in clinical deployment. Your MC Dropout approach produces calibration-relevant outputs (predictive entropy, confidence histograms). Reporting ECE alongside QWK and AUC would distinguish your thesis from nearly all existing work and directly address this gap.

**Gap 3 — Uncertainty-aware referral thresholding.** Papers that implement uncertainty estimation (Bayesian CNNs, MC Dropout) exist (Akram 2025 uses Bayesian approaches, your thesis uses MC Dropout). But no paper derives and clinically validates a specific entropy threshold for "refer to human grader" decisions, then reports what happens to sensitivity/specificity at that threshold. Your thesis is positioned to close this gap if you run the referral threshold analysis described in your project plan (Section 6.4).

**Gap 4 — Cross-dataset validation of attention mechanisms.** IDANet reports impressive cross-dataset results (DDR, EyePACS, IDRiD) but does not ablate whether CBAM/dual-attention is responsible for generalization vs. the training data mix. MediDRNet does not report Messidor-2 results. There is no paper that directly ablates CBAM on/off across multiple datasets with Messidor-2 as the external test. Your thesis design (train APTOS, test Messidor-2, ablate baseline vs. CBAM-ResNet50) would directly address this.

---

## Critical evaluation of your thesis architecture (dr-evaluate protocol)

---

**Idea: CBAM-ResNet50 + MC Dropout + Focal Loss for 5-class DR grading**

**Problem addressed** Three simultaneous limitations of baseline ResNet-50 for DR: (1) uniform processing of all image regions rather than attending to lesion areas, (2) point estimates with no uncertainty quantification, (3) class imbalance bias toward No DR (grade 0).

**Prior art**

- CBAM attention on CNN for medical imaging: Woo et al. (2018) ECCV — T3 evidence (natural images). MediDRNet (Teng 2024) uses CBAM specifically for DR — T1 evidence. IDANet (Bhati 2024) uses dual attention analogous to CBAM for DR — T1 evidence.
- MC Dropout for Bayesian uncertainty: Gal & Ghahramani (2016) — T3 evidence. Akram et al. (2025) applies Bayesian approaches to DR specifically — T1 evidence.
- Focal Loss for class imbalance: Lin et al. (2017) — T3 evidence (object detection). Romero-Oraá (2024) uses Focal Loss for DR — T1 evidence.

**Mechanism analysis** CBAM forces the network to weight lesion-containing spatial regions more heavily (spatial attention) and relevant feature channels more heavily (channel attention), which should reduce the influence of background retina on the classification decision. The evidence from MediDRNet (T1) confirms that CBAM specifically helps with subtle lesion features in DR — the mechanism is not just inherited from natural image experiments. MC Dropout works by sampling from an approximate posterior over network weights — high variance in predictions across T passes signals epistemic uncertainty. Focal Loss down-weights easy majority-class examples (grade 0) by a factor of up to 100×, forcing the gradient to concentrate on rare minority classes (grades 3 and 4).

**Potential failure modes**

1. APTOS label noise may interact badly with Focal Loss — Focal Loss amplifies gradients from hard examples, but some hard examples in APTOS are hard because they are mislabelled (single grader, Kaggle crowdsource), not because they are clinically ambiguous. `[Evidence: Chopra 2025 explicitly notes APTOS label noise as a key concern; Chopra 2025 Section IX warns that "label protocol variability can cause large AUC swings".]`

2. MC Dropout with p=0.5 on the final head only is a weak Bayesian approximation. Dropout on the ResNet backbone features would sample more diverse representations. With p=0.5 on a single Linear(2048→5) layer, variance across T=20 passes may be low and entropy estimates may be poorly calibrated. `[Inference — no T1 evidence for this specific configuration in DR.]`

3. CBAM placed after each stage (4 positions) adds attention at multiple scales — but the stages operate on very different spatial resolutions (128×128 down to 16×16). Spatial attention at layer4 (16×16 feature maps) has very coarse granularity — a single attention cell corresponds to 32×32 pixels in the original 512×512 image. Microaneurysms are typically 10–15px in a 512×512 fundus image, so layer4 spatial attention may be too coarse to localise them. `[Inference, grounded in known fundus lesion sizes from IDRiD annotations.]`

4. Focal Loss requires alpha weights derived from class frequencies. These weights are typically computed from the training set. With 5-fold stratified CV, each fold's training set has slightly different class frequencies — you should recompute alpha per fold, not fix it globally. If alpha is fixed from the full dataset, fold-specific imbalance is not fully corrected.

**Dataset-specific notes**

On APTOS 2019: the known label noise means that QWK is a more reliable primary metric than accuracy — QWK penalises predictions proportionally to how wrong they are on the ordinal scale, which provides partial robustness to single-grader noise. Stratified 5-fold CV is the correct split strategy. Your project correctly prioritises QWK.

On Messidor-2: adjudicated labels are cleaner. The domain shift from APTOS (Indian hospital fundus cameras) to Messidor-2 (French hospital cameras) is substantial. Expected performance drop: 5–15 QWK points (consistent with literature pattern Alwakid 2023: APTOS ~0.98 acc → DDR ~0.80 acc). Your project-overview.md correctly anticipates this and frames it as a discussion point — keep this framing.

**Verdict: Supported**

The combination is well-grounded. CBAM for DR is T1-supported via MediDRNet and IDANet. MC Dropout for uncertainty in medical imaging is T1-supported via Akram 2025. Focal Loss for DR class imbalance is T1-supported via Romero-Oraá 2024. No single paper combines all three in your configuration, which is the novel contribution. The most significant risk is the MC Dropout calibration (failure mode 2) — this is addressable by computing ECE.

**Recommended experiment** The highest-value ablation is: baseline ResNet-50 (CE loss) → baseline ResNet-50 (Focal Loss) → CBAM-ResNet50 (Focal Loss) → CBAM-ResNet50 (Focal Loss + MC Dropout). Each step isolates one contribution. Run all four on APTOS fold 0 and report QWK, AUC, and ECE. This is implementable in your existing codebase with minimal changes and is publishable as an ablation table.

---

## Summary: what this means for your thesis

The field's current gap is not raw performance (Swin-L and TOViT already push above 99% on APTOS) — it is **clinical trustworthiness**: calibrated uncertainty, reproducibility, and cross-domain generalization. Your thesis addresses exactly the right gaps for a 2026 undergraduate thesis: a reproducible, uncertainty-aware pipeline with external validation on Messidor-2 and calibration metrics. That is more valuable than chasing APTOS accuracy records that likely reflect dataset-specific overfitting.

The three concrete actions that would most strengthen your thesis given this literature review: (1) report ECE alongside QWK and AUC — almost no existing paper does this and it closes Gap 2 directly; (2) add the ordinal loss ablation as a supplementary experiment — it closes Gap 1 with minimal engineering overhead; (3) frame your Messidor-2 uncertainty analysis around a specific referral threshold with sensitivity/specificity at that threshold — it closes Gap 3 and makes the clinical contribution concrete.
