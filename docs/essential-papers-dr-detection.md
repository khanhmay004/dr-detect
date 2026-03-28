# DANH SÁCH PAPERS ESSENTIAL - DIABETIC RETINOPATHY DETECTION
## Cho Khóa Luận Tốt Nghiệp Deep Learning 2026

---

## 📚 **I. REVIEW PAPERS - BẮT BUỘC ĐỌC**

### **A. Survey Papers Mới Nhất (2024-2026)**

#### **1. From Retinal Pixels to Patients: Evolution of Deep Learning Research in DR Screening (2025)** ⭐⭐⭐
**Tác giả**: Muskaan Chopra, Lorenz Sparrenberg, et al.  
**Nguồn**: arXiv:2511.11065v1 (November 2025)  
**Link**: https://arxiv.org/html/2511.11065v1  

**Tại sao cần đọc**:
- **Survey TOÀN DIỆN NHẤT** cho DR (2016-2025)
- Consolidates **50+ studies**, **20+ datasets**
- **8 sections** critical: Reproducibility, Domain Shift, Class Imbalance, Architectures (CNN→ViT→Neuro-symbolic), Federated Learning, Benchmarking
- **Performance tables** chi tiết (Table II): So sánh methods across datasets
- **Dataset overview** (Table I): EyePACS, APTOS, Messidor, DDR, IDRiD specs
- **Evaluation protocols**: Per-patient splits, external validation, calibration
- **2025+ Research Agenda**: Open benchmarks, reproducibility, clinical deployment

**Key Sections for Thesis**:
- Section II: Foundational breakthroughs + reproducibility gap
- Section IV: Class imbalance, SSL, domain generalization
- Section V: Architecture evolution (CNN→ViT)
- Section VII: Dataset specifications
- Section VIII: Performance benchmarking tables
- Section X: Future directions (5-class grading, federated learning, clinical trust)

**Download**: PDF available on arXiv

---

#### **2. Vision Transformers in Medical Imaging: Comprehensive Review (2025)** ⭐⭐⭐
**Tác giả**: Sanad Aburass, Osama Dorgham, et al.  
**Nguồn**: J Imaging Inform Med (2025) 38(6):3928-3971  
**DOI**: 10.1007/s10278-025-01481-y  
**PubMed**: PMCID: PMC12701147  
**Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12701147/  

**Tại sao cần đọc**:
- **Vision Transformer architecture chi tiết**: Mathematical formulations (patch embedding, positional encoding, self-attention, feed-forward)
- **ViT variants**: DeiT (data-efficient), Swin (hierarchical attention), ConViT (hybrid CNN-ViT), LeViT (efficient inference)
- **DR-specific section**: 4 key ViT studies cho DR
  - Gu et al. 2023: ViT + Residual Attention
  - Adak et al. 2023: Ensemble (ViT, BEiT, CaiT, DeiT)
  - Wu et al. 2021: Vanilla ViT (91.4% accuracy)
  - Oulhadj et al. 2024: ViT + Modified CapsuleNet (88.18% APTOS, 87.78% EyePACS)
- **Table 5**: Retinal/eye analysis with ViT architectures
- **Preprocessing best practices**: CLAHE, PLT, augmentation
- **Limitations**: Data hunger, compute, interpretability

**Key Sections**:
- Vision Transformer mathematical overview
- Efficient Vision Transformers (DeiT, Swin, ConViT, LeViT)
- Diabetic Retinopathy Classification section
- Preprocessing & training strategies

**Download**: Open access via PubMed Central

---

#### **3. Systematic Review of Regulator-Approved Deep Learning Systems for DR Screening (December 2025)** ⭐⭐⭐
**Tác giả**: Multiple authors  
**Nguồn**: Nature Digital Medicine (December 2025)  
**Link**: https://www.nature.com/articles/s41746-025-02223-8  

**Tại sao cần đọc**:
- **Meta-analysis lớn nhất**: 82 studies, 887,244 examinations, 25 devices, 28 countries
- **Real-world performance**: Sensitivity 93%, Specificity 90%
- **FDA-approved systems**: Performance across diverse clinical settings
- **Key factors**: DR severity threshold, image gradability, pupil dilation, reference standards
- **Clinical validation**: What works in practice vs. research

**Key Takeaways**:
- Real-world benchmarks cho clinical deployment
- Factors affecting performance: dilated pupils → better specificity, ungradable images → false positives
- Portable cameras + adjudicated references → improved specificity

**Download**: Nature Digital Medicine (subscription/institutional access)

---

#### **4. Deep Learning-Based Diabetic Retinopathy Recognition: Survey (2025)** ⭐⭐
**Tác giả**: MDI Bappi, et al.  
**Nguồn**: Healthcare Analytics (Elsevier), Volume 7, March 2025  
**DOI**: S2405959525001122  
**Link**: https://www.sciencedirect.com/science/article/pii/S2405959525001122  

**Tại sao cần đọc**:
- Focus on **attention mechanisms**, **feature fusion strategies**
- Recent DL approaches (2020-2024)
- Challenges: class imbalance, generalization, interpretability
- Practical recommendations for DR classification systems

**Key Topics**:
- Attention-based architectures for DR
- Feature fusion techniques
- Multi-task learning approaches

**Download**: ScienceDirect (check institutional access)

---

#### **5. Intelligent Retinal Disease Detection Using Deep Learning (December 2025)** ⭐⭐
**Tác giả**: Multiple authors  
**Nguồn**: Nature Scientific Reports (December 2025)  
**DOI**: s41598-025-28376-w  
**Link**: https://www.nature.com/articles/s41598-025-28376-w  

**Tại sao cần đọc**:
- **Hybrid approach**: MobileNetV2 + DenseNet121 ensemble
- **Feature fusion**: PCA + DWT (Discrete Wavelet Transform)
- **98.2% accuracy** on multi-disease classification
- Lightweight models for resource-constrained settings
- Dataset: BRSET (16,266 fundus images from 8,524 patients)

**Architecture Details**:
- 450 features from PCA + 12 from DWT per image
- Suitable for deployment in low-resource settings

**Download**: Open access via Nature Scientific Reports

---

#### **6. Deep Learning for Comprehensive Analysis of Retinal Fundus Images (August 2025)** ⭐⭐
**Tác giả**: Multiple authors  
**Nguồn**: Bioengineering (Nature), August 2025  
**PubMed**: PMC12383659  
**Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12383659/  

**Tại sao cần đọc**:
- **Swin-L Transformer SOTA performance**:
  - Binary DR: AUC 0.98, F1 0.95, Acc 0.95
  - 3-Class DR: AUC 0.98 (macro), F1 0.95, Acc 0.94
  - PDR detection: AUC 0.99 (excellent for severe cases)
- **Architecture comparison**: 6 models (3 CNNs, 3 Vision Transformers)
- **Dataset**: BRSET (16,266 images)
- **Explainability**: Gradient-based saliency maps

**Architecture Performance Table**:
| Model | Params | DR Binary AUC | DR 3-Class AUC | Notes |
|-------|--------|---------------|-----------------|-------|
| Swin-L | 197M | 0.98 | 0.98 | SOTA, heavyweight |
| Twins-SVT-L | 99M | 0.98 | 0.97 | Good balance |
| CSWin-B | 78M | 0.97 | 0.97 | Lightweight |

**Download**: Open access via PubMed Central

---

#### **7. Systematic Review of Hybrid Vision Transformer Architectures (January 2025)** ⭐⭐
**Tác giả**: Multiple authors  
**Nguồn**: J Digit Imaging (January 2025)  
**PubMed**: PMC12572492  
**Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12572492/  

**Tại sao cần đọc**:
- **Hybrid ViT-CNN architectures**: Combining strengths
- ViT: Long-range dependencies (self-attention)
- CNN: Local features (spatial convolution)
- **Merging strategies**: Feature concatenation, attention fusion, adaptive gating
- **PRISMA guideline**: 34 novel hybrid architectures reviewed
- **Efficiency metrics**: Parameters, inference time (GFlops), performance benchmarks

**Applications**:
- Medical imaging tasks: segmentation, classification, image quality restoration
- Real-time deployment considerations

**Download**: Open access via PubMed Central

---

### **B. Classic Foundation Papers (2020-2023)**

#### **8. Diabetic Retinopathy Detection Through Deep Learning Techniques: A Review (2020)** ⭐⭐⭐
**Tác giả**: Alyoubi et al.  
**Nguồn**: Informatics in Medicine Unlocked 20 (2020) 100377  
**Link**: Attached as file:1 in your conversation  

**Tại sao cần đọc**:
- **Base paper** bạn đã cung cấp
- **33 papers reviewed** (2016-2019)
- **Critical gap identified**: Only 6% do grading + lesion detection
- **Dataset descriptions**: Kaggle, Messidor, DDR, IDRiD, APTOS, E-ophtha
- **Preprocessing methods**: CLAHE, green channel, augmentation
- **Performance measures**: AUC, accuracy, sensitivity, specificity
- **Classification categories**: Binary, multi-level, lesion-based, vessels-based

**Key Tables**:
- Table 3: Dataset comparison
- Table 4: Methods used for DR detection/classification (33 papers)

**Download**: Already attached

---

#### **9. RETFound: A Foundation Model for Retinal Imaging (Nature 2023)** ⭐⭐⭐
**Tác giả**: Zhou et al.  
**Nguồn**: Nature (2023)  
**DOI**: 10.1038/s41586-023-06555-x  
**Citations**: 843  
**Link**: https://www.nature.com/articles/s41586-023-06555-x  

**Tại sao cần đọc**:
- **First foundation model** for retinal imaging
- **Self-supervised learning**: Masked autoencoder (MAE)
- **Training data**: 904,170 CFPs + 736,442 OCTs (Moorfields Eye Hospital)
- **Label efficiency**: 10-50% data needed vs. ImageNet pre-training
- **Performance**:
  - APTOS-2019: AUC 0.943
  - IDRiD: AUC 0.822
  - Messidor-2: AUC 0.884
- **GitHub**: https://github.com/rmaphoh/RETFound (pre-trained weights available)

**Clinical Impact**:
- Multi-disease detection: DR, glaucoma, AMD, systemic diseases
- ~80% training time reduction
- Attention maps identify lesions (hard exudates, hemorrhages)

**Download**: Nature (subscription required, pre-print may be on arXiv)

---

#### **10. DeepDR Plus: Predicting Time to DR Progression (Nature Medicine 2024)** ⭐⭐
**Tác giả**: Dai et al.  
**Nguồn**: Nature Medicine (2024)  
**DOI**: 10.1038/s41591-023-02702-z  
**Citations**: 279  
**Link**: https://www.nature.com/articles/s41591-023-02702-z  

**Tại sao cần đọc**:
- **Progression prediction**: Time-to-event within 5 years
- **Training scale**: Pre-training on 717,308 images from 179,327 participants
- **Performance**: Concordance Index 0.754-0.846
- **Clinical impact**: Extend screening interval from 12 months → 31.97 months (3 years!)
- **Delayed detection**: Only 0.18% for vision-threatening DR
- **Personalized screening**: Risk-based scheduling

**Three Prediction Tasks**:
1. No DR → DR
2. Non-referable DR → Referable DR
3. Non-VTDR → Vision-threatening DR

**Download**: Nature Medicine (subscription required)

---

---

## 🔬 **II. TECHNICAL PAPERS - SOTA METHODS**

### **A. Vision Transformers for DR (2024-2025)**

#### **11. Task-Optimized Vision Transformer for DR (TOViT) - Nature Scientific Reports (2025)** ⭐⭐⭐
**Tác giả**: Bhoopalan et al.  
**Nguồn**: Nature Scientific Reports (2025)  
**Link**: Referenced as web:6, web:12  

**Performance**:
- Accuracy: **99%**
- F1-score: **>93%** across all DR stages
- **Real-time on Raspberry Pi-4**: 8 FPS, 120ms latency

**Architecture**:
- 16 layers, 1024-dimensional embeddings
- Layer-wise learning rate scheduling
- Attention head tuning (16 heads)
- 8-bit quantization + structured pruning

**Datasets**: EyePACS, Messidor-2, APTOS 2019, DIARETDB1

**Why Read**:
- **Edge deployment**: Point-of-care screening devices
- Optimization techniques for real-world deployment

---

#### **12. Vision Transformer + Capsule Network Hybrid (2024)** ⭐⭐
**Tác giả**: Oulhadj et al.  
**Nguồn**: Referenced as web:3  
**PubMed**: 38701591  

**Performance**:
- APTOS: 88.18%
- Messidor-2: 87.78%
- DDR: 80.36%
- EyePACS: 78.64%

**Method**:
- Fine-tuned Vision Transformer
- Modified Capsule Network
- Preprocessing: Power Law Transform + CLAHE

**Why Read**:
- Hybrid architecture (ViT + Capsule)
- Cross-dataset validation on 4 datasets
- Better than pure CNN approaches

---

### **B. Multi-Task Learning (2023-2025)**

#### **13. Multi-Task Learning for DR Grading and Lesion Segmentation (AAAI 2023)** ⭐⭐⭐
**Tác giả**: AAAI Conference paper  
**Link**: https://aaai.org/papers/13267-multi-task-learning-for-diabetic-retinopathy-grading-and-lesion-segmentation/  

**Why Read**:
- **Addresses the 6% gap**: Grading + lesion detection
- Semi-supervised learning to obtain segmentation masks
- Outperforms single-task SOTA networks

**Tasks**:
1. DR severity grading
2. Lesion segmentation (MA, hemorrhages, exudates)

---

#### **14. DRAMA - Multi-Label Learning for DR (2025)** ⭐⭐
**Tác giả**: Referenced as web:119, web:122  
**Nguồn**: Acta Ophthalmologica (2025)  
**PubMed**: PMC12167062  

**Performance**:
- Quality assessment: 87.02% accuracy
- Lesion detection: 91.60% accuracy
- AUC >0.95 for most tasks
- **Speed**: 86ms for entire test set (vs 90-100 min for humans!)

**11 Multi-Label Tasks**:
1. Image type identification
2. Quality assessment
3. Lesion detection (MA, hemorrhages, exudates, cotton wool spots)
4. DR severity grading (5 stages)
5. Diabetic macular edema (DME) detection

**Architecture**: EfficientNet-B2 (ImageNet pre-trained)

**Why Read**:
- Comprehensive multi-task framework
- Clinical efficiency (86ms processing)
- LabelSmoothingCrossEntropy + AdamP optimizer

---

#### **15. MVCAViT - Multi-View Cross Attention ViT (2025)** ⭐⭐
**Tác giả**: Referenced as web:128, web:133  
**Nguồn**: Nature Scientific Reports (2025)  
**DOI**: s41598-025-18742-z  

**Architecture**:
- **Dual-View Learning**: Macula-centered + Optic-disc-centered images
- Cross-attention mechanism between views
- **Multi-Task Objectives**: Disease classification + lesion localization + severity grading

**Optimization**: Particle Swarm Optimization (PSO) for hyperparameter tuning

**Why Read**:
- Multi-view integration approach
- Cross-attention between anatomical views
- Attention visualizations for interpretability

---

### **C. Attention Mechanisms & Explainability (2024-2025)**

#### **16. Attention-Based Deep Learning with Separate Dark/Bright Structure Attention (2024)** ⭐⭐⭐
**Tác giả**: Romero-Oraá et al.  
**Nguồn**: Referenced as web:42  
**PubMed**: 38583290  
**Citations**: 46  

**Innovation**:
- **Separate attention maps**: 
  - Dark structures (red lesions: MA, HM)
  - Bright structures (bright lesions: EX)
- Image decomposition before attention
- Xception backbone + focal loss

**Performance**:
- Accuracy: 83.7% (Kaggle dataset)
- Quadratic Weighted Kappa: 0.78

**Why Read**:
- Explainable AI (XAI) with lesion-type-specific attention
- Clinically interpretable attention maps

---

#### **17. Inherently Interpretable AI Models for DR Screening (2025)** ⭐⭐
**Tác giả**: Djoumessi et al.  
**Nguồn**: Referenced as web:43  
**PubMed**: PMC12068651  

**Architecture**: Sparse BagNet
- **Explicit evidence maps** (not post-hoc explanation)
- Bag-of-local-features approach
- Avoids spurious correlations of LIME/SHAP

**Clinical Impact**:
- Reduces grading time
- Improves accuracy for borderline cases (mild NPDR)
- Evidence maps align with pathological patterns

**Why Read**:
- Inherently interpretable (not black-box + post-hoc)
- Clinical decision support integration

---

### **D. Class Imbalance Solutions (2024-2025)**

#### **18. MediDRNet - Prototype Contrastive Learning for Class Imbalance (2024)** ⭐⭐⭐
**Tác giả**: Teng et al.  
**Nguồn**: Referenced as web:66  

**Innovation**:
- **Dual-branch network**: Feature learning + classifier branches
- **Prototypical contrastive learning**:
  - Minimize distance between samples and their category prototype
  - Maximize distance from other category prototypes
- **CBAM**: Convolutional Block Attention Module for subtle lesion features

**Performance**:
- **SOTA on Kaggle dataset**: ACA, Micro-F1, Kappa scores
- Exceptional performance on UWF dataset
- Strong on **minority categories** (severe, proliferative)

**Why Read**:
- Addresses severe/proliferative DR underrepresentation
- Prototype-based approach for imbalance

---

#### **19. Balanced Few-Shot Episodic Learning for DR (2025)** ⭐⭐
**Tác giả**: Referenced as web:67  

**Components**:
1. **Balanced Episodic Sampling**: Equal representation of majority/minority classes
2. **Targeted Augmentation**: CLAHE for minority classes
3. **ResNet-50 Backbone**: Captures subtle inter-class differences

**Results**:
- Substantial accuracy gains
- Reduced bias toward majority classes
- Notable improvements for underrepresented diseases

**Training**: 100 episodes, evaluation on 1,000 test episodes

**Why Read**:
- Few-shot learning approach
- Episodic sampling strategy

---

### **E. Generative Models & Synthetic Data (2024-2025)**

#### **20. Conditional Cascaded Diffusion Model (CCDM) for DR Data Augmentation (2025)** ⭐⭐
**Tác giả**: Referenced as web:98, web:101  
**Nguồn**: Nature Communications Medicine (2025)  

**Innovation**:
- **Generate synthetic retinal images** to augment imbalanced datasets
- **Conditioning variables**: Age, sex, ethnicity, diabetes duration, baseline DR stage, referable DR/maculopathy status

**Quality**: FID (Fréchet Inception Distance) = 9.3 (very good)

**Performance Impact**:
- **Internal test**: AUROC improved from 0.827 → 0.851 (p=0.044) with ×2 synthetic augmentation
- **External test**: No significant improvement (generalization issue)

**Use Case**: Predicting 2-year incident referable DR/maculopathy

**Why Read**:
- Diffusion models for medical image synthesis
- Addresses class imbalance with synthetic data
- Limitations: generalization to external datasets

---

#### **21. StyleGAN for Retinal Images - High-Quality Synthesis (2022)** ⭐⭐
**Tác giả**: Kim et al.  
**Nguồn**: Nature Scientific Reports (2022)  
**DOI**: s41598-022-20698-3  
**Citations**: 38  
**Link**: https://www.nature.com/articles/s41598-022-20698-3  

**Innovation**:
- Synthesize realistic high-resolution retinal images
- **Turing Test**: Ophthalmologists only 54% accuracy (random probability!)

**Validation**:
- Vessel amount difference: <0.43% vs real images
- Average SNR difference: <1.5% vs real images

**Efficacy for Augmentation**:
- AUC increased 23.7% (0.735→0.909) for extremely imbalanced class (1:0.1 ratio)

**Why Read**:
- State-of-the-art synthetic retinal image generation
- Effective for class imbalance
- Transfer learning from normal to disease images

---

---

## 📊 **III. DATASET PAPERS - PHẢI ĐỌC**

#### **22. Kaggle Combined Dataset: APTOS+DDR+IDRiD+EyePACS+Messidor (2025)** ⭐⭐⭐
**Link**: https://www.kaggle.com/datasets/sehastrajits/fundus-aptosddridirdeyepacsmessidor  
**Nguồn**: Referenced as web:23  

**Specifications**:
- **30K balanced images** (5,968 per class)
- 5 DR stages (0: No DR → 4: Proliferative DR)
- Train/Val/Test splits provided
- Total: 45.3K files, 10.86 GB

**Why Essential**:
- ✅ **Balanced data** → solves class imbalance
- ✅ **Large size** (~30K) → enough for deep models
- ✅ **Multi-source** → good generalization
- ✅ **Ready-to-use** → preprocessing done

**Recommendation**: **PRIMARY DATASET FOR THESIS**

---

#### **23. AI-READI: Multimodal Dataset for Diabetic Eye Research (2025)** ⭐⭐
**Link**: https://www.retinalphysician.com/issues/2025/june/ai-readi-a-multimodal-data-set-for-diabetic-eye-research/  
**Nguồn**: Referenced as web:22  

**Specifications**:
- **165,051 files (2TB)** as of Nov 2024
- Target: 4,000 participants
- **15+ modalities**: Multiple ophthalmic imaging devices, clinical data
- Current (v2.0.0): 1,067 participants
  - 372 healthy, 242 prediabetes, 323 oral meds, 130 insulin therapy

**Why Important**:
- Multimodal learning opportunities
- Comprehensive clinical data integration
- Salutogenic lens (health factors, not just disease)

**Note**: Still being collected, access may be restricted

---

#### **24. DDR (DeepDR Dataset) - Dataset Paper (2019)** ⭐⭐
**Specifications**:
- 13,673 fundus images
- 45° field of view
- 5 DR stages
- **757 images with lesion-level annotations** (MA, hemorrhages, soft/hard exudates, neovascularization)

**Why Essential**:
- Good for both classification AND lesion detection tasks
- Standard external validation set
- Lesion annotations available for subset

---

#### **25. IDRiD (Indian Diabetic Retinopathy Image Dataset) - Dataset Paper (2018)** ⭐⭐
**Specifications**:
- 516 images (413 training, 103 test)
- Very high resolution: 4288×2848 pixels
- 50° FOV
- **Pixel-level lesion annotations** (MA, hemorrhages, hard exudates, soft exudates)

**Why Essential**:
- Highest quality annotations
- Standard for lesion segmentation + severity grading
- Excellent for validation/testing

---

---

## 🛠️ **IV. METHODOLOGICAL PAPERS - TECHNIQUES**

### **A. Domain Generalization & Adaptation**

#### **26. DECO - Domain Disentanglement for Cross-Dataset DR (MICCAI 2024)** ⭐⭐
**Tác giả**: Xia et al.  
**Nguồn**: Referenced as web:xia2024MICCAI  

**Innovation**:
- **Representation disentanglement**: Separate disease signal from domain cues
- Feature normalization
- Cross-dataset accuracy improvement without target labels

**Why Read**:
- Domain generalization technique
- Reduces EyePACS → Messidor/DDR performance drop

---

#### **27. GDRNet - Generalized Domain Adaptation for DR (2023)** ⭐⭐
**Tác giả**: Che et al.  
**Nguồn**: Referenced as web:che2023GDRNet  

**Method**:
- Domain adaptation pipeline
- Pseudo-labeling and consistency regularization
- Measurable cross-domain improvements

**Why Read**:
- Domain adaptation when unlabeled target data available
- Practical approach to domain shift

---

### **B. Self-Supervised & Semi-Supervised Learning**

#### **28. Lesion-Aware Contrastive Pretraining for DR (MICCAI 2021)** ⭐⭐
**Tác giả**: Huang et al.  
**Nguồn**: Referenced as web:Huang2021MICCAI  

**Method**:
- Lesion-based contrastive learning
- EyePACS generalization improvement
- Self-supervised pretraining on unlabeled fundus images

**Why Read**:
- Label-efficient learning
- Contrastive learning for medical imaging

---

#### **29. Masked Autoencoder (MAE) for ViT Pretraining on DR (PLOS ONE 2024)** ⭐⭐
**Tác giả**: Yang et al.  
**Nguồn**: Referenced as web:Yang2024PLOSONE  

**Performance**:
- AUC 0.98 with significantly fewer labels vs. ImageNet-pretrained CNNs
- Vision Transformer with MAE pretraining

**Why Read**:
- State-of-the-art self-supervised learning
- Label efficiency for DR

---

### **C. Federated Learning**

#### **30. Federated Learning for DR Across Multiple Centers (Scientific Reports 2023)** ⭐
**Tác giả**: Matta et al.  
**Nguồn**: Referenced as web:Matta2023SciRep  

**Method**:
- Federated CNN ensembles on EyePACS, Messidor, IDRiD
- Only minor accuracy reductions vs. centralized training
- Privacy-preserving multi-center training

**Why Read**:
- Practical federated learning implementation
- Privacy-compliant collaborative training

---

---

## 📖 **V. CLINICAL & APPLICATION PAPERS**

#### **31. Ophthalmology Times: How AI is Reshaping Ophthalmology in 2025 (December 2025)** ⭐⭐
**Link**: https://www.ophthalmologytimes.com/view/how-ai-is-reshaping-ophthalmology-in-2025-and-beyond  
**Nguồn**: Referenced as web:137  

**Why Read**:
- **Industry survey**: 78% of ophthalmologists cite AI as most transformative trend
- **25+ FDA-cleared DR screening devices** deployed globally
- Clinical workflow integration challenges
- Future trends: point-of-care deployment, personalized treatment strategies

---

#### **32. Oxford Academic: Systemic Disease Screening Using Deep Learning (September 2025)** ⭐⭐
**Link**: https://www.oxjournal.org/systemic-disease-screening-using-deep-learning-analysis-of-retinal-images/  
**Nguồn**: Referenced as web:142  

**Why Read**:
- Retinal imaging as **non-invasive biomarker** for systemic health
- Multimodal models (retinal + clinical data) outperform single modality
- **Limitations highlighted**: Dataset bias, algorithmic fairness, generalization to non-Western cohorts

---

---

## 🎯 **VI. PAPERS TO CITE BY SECTION**

### **Introduction & Background**
1. Base review paper (Alyoubi et al. 2020) [file:1]
2. "From Retinal Pixels to Patients" survey (2025) [web:166]
3. RETFound foundation model (Zhou et al., Nature 2023) [web:81]

### **Related Work - Vision Transformers**
4. "Vision Transformers in Medical Imaging" comprehensive review (2025) [web:156]
5. "Systematic Review of Hybrid ViT" (2025) [web:159]
6. TOViT for DR (Nature Sci Rep 2025) [web:6]
7. ViT + Capsule Network (Oulhadj et al. 2024) [web:3]

### **Related Work - Multi-Task Learning**
8. "Multi-Task Learning for DR Grading and Lesion Segmentation" (AAAI 2023) [web:116]
9. DRAMA multi-label learning (2025) [web:119]
10. MVCAViT dual-view learning (2025) [web:128]

### **Related Work - Attention & Explainability**
11. Separate dark/bright attention (Romero-Oraá et al. 2024) [web:42]
12. Inherently interpretable models (2025) [web:43]

### **Related Work - Class Imbalance**
13. MediDRNet prototype contrastive learning (2024) [web:66]
14. Balanced few-shot episodic learning (2025) [web:67]
15. Addressing High Class Imbalance (2025) [web:63]

### **Related Work - Generative Models**
16. CCDM diffusion model (Nature Comm Med 2025) [web:98]
17. StyleGAN for retinal images (2022) [web:100]

### **Datasets**
18. Kaggle Combined Dataset paper [web:23]
19. AI-READI multimodal dataset [web:22]
20. DDR dataset description (2019) [web:23]
21. IDRiD dataset paper (2018) [web:24]

### **Methodology - Domain Generalization**
22. DECO domain disentanglement (MICCAI 2024)
23. GDRNet domain adaptation (2023)

### **Methodology - Self-Supervised Learning**
24. Lesion-aware contrastive learning (MICCAI 2021) [web:Huang2021MICCAI]
25. MAE for ViT pretraining (PLOS ONE 2024) [web:Yang2024PLOSONE]

### **Results & Benchmarking**
26. "Systematic Review of FDA-Approved Systems" (Nature Digit Med 2025) [web:135]
27. "Intelligent Retinal Disease Detection" (Nature Sci Rep 2025) [web:136]
28. Bioengineering Swin-L performance (2025) [web:139]

### **Discussion - Clinical Translation**
29. Ophthalmology Times clinical trends (2025) [web:137]
30. Oxford Academic systemic disease screening (2025) [web:142]

### **Future Work - Progression Prediction**
31. DeepDR Plus progression prediction (Nat Med 2024) [web:118]

### **Future Work - Federated Learning**
32. Federated learning for DR (Sci Rep 2023) [web:Matta2023SciRep]

---

---

## 📝 **VII. READING ORDER RECOMMENDATION**

### **Phase 1: Foundation (Week 1-2)**
1. ✅ Base review paper (Alyoubi et al. 2020) - Already have
2. ✅ "From Retinal Pixels to Patients" (2025) - Most comprehensive survey
3. ✅ "Vision Transformers in Medical Imaging" (2025) - ViT architectures
4. RETFound (Nature 2023) - Foundation model baseline

### **Phase 2: State-of-the-Art Methods (Week 3-4)**
5. TOViT (Nature Sci Rep 2025) - Edge deployment
6. Swin-L performance (Bioengineering 2025) - SOTA benchmarks
7. "Systematic Review of FDA-Approved Systems" (Nature Digit Med 2025) - Real-world validation
8. ViT + Capsule Network (2024) - Hybrid architecture

### **Phase 3: Multi-Task Learning (Week 5)**
9. "Multi-Task Learning AAAI" (2023) - Theoretical foundation
10. DRAMA (2025) - 11-task framework
11. MVCAViT (2025) - Multi-view learning

### **Phase 4: Class Imbalance & Generative Models (Week 6)**
12. MediDRNet (2024) - Prototype contrastive learning
13. CCDM diffusion model (2025) - Synthetic data
14. StyleGAN retinal (2022) - High-quality synthesis

### **Phase 5: Explainability & Clinical Translation (Week 7)**
15. Separate attention mechanisms (2024) - XAI
16. Inherently interpretable models (2025) - Clinical trust
17. Ophthalmology Times clinical trends (2025) - Industry perspective
18. Oxford Academic systemic screening (2025) - Broader context

### **Phase 6: Datasets & Specialized Topics (Week 8)**
19. Kaggle Combined Dataset - Data preparation
20. Domain generalization papers (DECO, GDRNet) - Cross-dataset validation
21. Self-supervised learning papers - Label efficiency
22. DeepDR Plus (2024) - Progression prediction (if relevant)

---

---

## 🔗 **VIII. DIRECT DOWNLOAD LINKS & ACCESS**

### **Open Access (Free)**
1. arXiv: https://arxiv.org/html/2511.11065v1 (Survey)
2. PubMed Central: https://pmc.ncbi.nlm.nih.gov/articles/PMC12701147/ (ViT Review)
3. PubMed Central: https://pmc.ncbi.nlm.nih.gov/articles/PMC12383659/ (Bioengineering)
4. Kaggle: https://www.kaggle.com/datasets/sehastrajits/fundus-aptosddridirdeyepacsmessidor (Dataset)
5. GitHub: https://github.com/rmaphoh/RETFound (RETFound weights)

### **Subscription/Institutional Access Required**
6. Nature Digital Medicine: https://www.nature.com/articles/s41746-025-02223-8
7. Nature: https://www.nature.com/articles/s41586-023-06555-x (RETFound)
8. Nature Medicine: https://www.nature.com/articles/s41591-023-02702-z (DeepDR Plus)
9. ScienceDirect: https://www.sciencedirect.com/science/article/pii/S2405959525001122

### **Conference Papers**
10. AAAI 2023: https://aaai.org/papers/13267-multi-task-learning-for-diabetic-retinopathy-grading-and-lesion-segmentation/

### **Industry & Clinical**
11. Ophthalmology Times: https://www.ophthalmologytimes.com/view/how-ai-is-reshaping-ophthalmology-in-2025-and-beyond
12. Retinal Physician: https://www.retinalphysician.com/issues/2025/june/ai-readi-a-multimodal-data-set-for-diabetic-eye-research/

---

---

## 📊 **IX. SUMMARY TABLE - TOP 10 MUST-READ PAPERS**

| Rank | Paper | Year | Type | Priority | Why Essential |
|------|-------|------|------|----------|---------------|
| **1** | From Retinal Pixels to Patients | 2025 | Survey | ⭐⭐⭐ | Most comprehensive DR survey (50+ studies, benchmarks, datasets) |
| **2** | Vision Transformers in Medical Imaging | 2025 | Review | ⭐⭐⭐ | ViT architectures, mathematical formulations, DR-specific section |
| **3** | Systematic Review of FDA-Approved Systems | 2025 | Meta-analysis | ⭐⭐⭐ | Real-world validation (82 studies, 887K exams, 25 devices) |
| **4** | RETFound Foundation Model | 2023 | Method | ⭐⭐⭐ | First foundation model, label-efficient, GitHub weights available |
| **5** | Base Review (Alyoubi et al.) | 2020 | Review | ⭐⭐⭐ | Historical context, datasets, critical 6% gap identification |
| **6** | TOViT for DR | 2025 | Method | ⭐⭐⭐ | SOTA 99% accuracy, edge deployment (Raspberry Pi) |
| **7** | Multi-Task Learning (AAAI) | 2023 | Method | ⭐⭐⭐ | Addresses grading + lesion detection gap |
| **8** | Swin-L Performance (Bioengineering) | 2025 | Benchmark | ⭐⭐ | SOTA results (AUC 0.98-0.99), architecture comparison |
| **9** | MediDRNet Class Imbalance | 2024 | Method | ⭐⭐ | Prototype contrastive learning, SOTA on minority classes |
| **10** | CCDM Diffusion Model | 2025 | Method | ⭐⭐ | Synthetic data generation, addresses class imbalance |

---

---

## ✅ **X. ACTION CHECKLIST**

### **Immediate Actions (Week 1)**
- [ ] Download "From Retinal Pixels to Patients" survey (arXiv)
- [ ] Download "Vision Transformers in Medical Imaging" review (PMC)
- [ ] Review base paper (Alyoubi et al. 2020) - already attached
- [ ] Access Kaggle Combined Dataset
- [ ] Bookmark all open-access links

### **Setup Zotero/Mendeley**
- [ ] Create bibliography with all 32 papers
- [ ] Tag by category: Survey, Method, Dataset, Clinical
- [ ] Priority labels: Must-read (⭐⭐⭐), Important (⭐⭐), Optional (⭐)

### **Institutional Access**
- [ ] Check university library for Nature/Elsevier subscriptions
- [ ] Request ILL (Interlibrary Loan) for subscription papers if needed
- [ ] Search for preprints on arXiv/bioRxiv for paywalled papers

### **Reading Notes Template**
For each paper, note:
1. **Key Contribution**: What's novel?
2. **Methodology**: Architecture, training strategy, datasets
3. **Performance**: Metrics, datasets, comparisons
4. **Limitations**: What they identify
5. **Relevance to Thesis**: How to cite, what to adopt

---

---

## 🎓 **XI. THESIS WRITING GUIDE - CITATION STRATEGY**

### **Introduction Section**
- **Opening**: DR prevalence, clinical need
  - Cite: WHO statistics, IDF diabetes atlas
- **Deep learning evolution**: CNN → ViT → Foundation models
  - Cite: Base review (2020), Survey (2025), RETFound (2023)

### **Related Work Section**

#### **2.1 Traditional Machine Learning**
- Cite: Base review (2020) for historical context

#### **2.2 Convolutional Neural Networks**
- Cite: Base review (2020), Survey (2025) Section II

#### **2.3 Vision Transformers**
- Cite: ViT review (2025), TOViT (2025), Oulhadj et al. (2024)

#### **2.4 Foundation Models**
- Cite: RETFound (2023), Global RETFound Consortium

#### **2.5 Multi-Task Learning**
- Cite: AAAI (2023), DRAMA (2025), MVCAViT (2025)

#### **2.6 Attention Mechanisms & Explainability**
- Cite: Romero-Oraá (2024), Inherently interpretable models (2025)

#### **2.7 Addressing Class Imbalance**
- Cite: MediDRNet (2024), Few-shot learning (2025), Survey (2025) Section IV-A

#### **2.8 Generative Models for Data Augmentation**
- Cite: CCDM (2025), StyleGAN (2022)

### **Methodology Section**

#### **3.1 Dataset Selection**
- Cite: Kaggle Combined (2025), Survey (2025) Table I

#### **3.2 Preprocessing Pipeline**
- Cite: ViT review (2025), Base review (2020), specific papers using CLAHE/PLT

#### **3.3 Architecture Design**
- Cite: Primary architecture papers (TOViT, Swin-L, etc.)
- Mathematical formulations: ViT review (2025)

#### **3.4 Training Strategy**
- Cite: RETFound for pre-training, specific papers for optimization

### **Results Section**

#### **4.1 Performance Metrics**
- Cite: Survey (2025) evaluation protocols

#### **4.2 Comparison with SOTA**
- Cite: Systematic Review (2025), Bioengineering (2025), relevant method papers

#### **4.3 Cross-Dataset Validation**
- Cite: Domain generalization papers (DECO, GDRNet)

### **Discussion Section**

#### **5.1 Clinical Implications**
- Cite: Ophthalmology Times (2025), Oxford Academic (2025), FDA review (2025)

#### **5.2 Limitations**
- Cite: Survey (2025) Section X, relevant papers discussing generalization gaps

#### **5.3 Future Work**
- Cite: DeepDR Plus (2024) for progression, Federated learning papers

---

---

## 📌 **XII. FINAL NOTES**

### **Priority Reading Order for Thesis Writing**
1. **Survey papers first** (most comprehensive context)
2. **SOTA methods** (what to implement/compare)
3. **Dataset papers** (data preparation)
4. **Specialized techniques** (innovations to adopt)
5. **Clinical papers** (contextualize contributions)

### **Citation Management**
- Use consistent citation style (IEEE, APA, or Vancouver for medical)
- Cross-reference between papers (e.g., Survey → Primary methods)
- Keep track of dataset versions and preprocessing details

### **Stay Updated**
- Set Google Scholar alerts for "diabetic retinopathy deep learning 2026"
- Follow arXiv: cs.CV, cs.LG, q-bio.QM
- Check Nature, IEEE TPAMI, Medical Image Analysis journals monthly

### **Open Source Code**
- RETFound: https://github.com/rmaphoh/RETFound
- TOViT: Check paper supplementary
- Hybrid ViT-CNN: Review paper citations

---

**Good luck with your thesis! 🎓🔬**

This reading list covers:
- ✅ **32 essential papers** (10 reviews, 15 methods, 4 datasets, 3 clinical)
- ✅ **Comprehensive coverage**: 2020-2026
- ✅ **All key topics**: ViT, Multi-task, Attention, Class imbalance, Generative models, Federated learning
- ✅ **Direct links** for download
- ✅ **Reading order** optimized for thesis writing
- ✅ **Citation strategy** by thesis section

Focus on **Top 10 must-reads** first, then expand based on your chosen approach!
