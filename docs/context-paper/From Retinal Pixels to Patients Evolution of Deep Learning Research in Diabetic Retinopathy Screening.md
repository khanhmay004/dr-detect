
From Retinal Pixels to Patients: Evolution of Deep Learning Research in Diabetic Retinopathy Screening

**Authors:** Muskaan Chopra, Lorenz Sparrenberg, Armin Berger, Sarthak Khanna, Jan H. Terheyden, Rafet Sifa **Affiliations:** Fraunhofer IAIS; University of Bonn; University Hospital Bonn; Lamarr Institute **Date:** November 2025

## Abstract

Diabetic Retinopathy (DR) remains a leading cause of preventable blindness, with early detection critical for reducing vision loss worldwide. Over the past decade, deep learning has transformed DR screening, progressing from early convolutional neural networks trained on private datasets to advanced pipelines addressing class imbalance, label scarcity, domain shift, and interpretability. This survey provides the first systematic synthesis of DR research spanning 2016-2025, consolidating results from 50+ studies and over 20 datasets. We critically examine methodological advances, including self- and semi-supervised learning, domain generalization, federated training, and hybrid neuro-symbolic models, alongside evaluation protocols, reporting standards, and reproducibility challenges. Benchmark tables contextualize performance across datasets, while discussion highlights open gaps in multi-center validation and clinical trust. By linking technical progress with translational barriers, this work outlines a practical agenda for reproducible, privacy-preserving, and clinically deployable DR AI. Beyond DR, many of the surveyed innovations extend broadly to medical imaging at scale.

Index Terms—Diabetic Retinopathy, Deep Learning, SelfSupervised Learning, Domain Generalization, Medical Imaging

---

## I. Introduction

Diabetic Retinopathy (DR) is a leading cause of preventable blindness; early detection is critical to reduce vision loss. According to the International Diabetes Federation (IDF), approximately 537 million adults (aged 20-79) are currently living with diabetes, and this number is projected to rise to 643 million by 2030 [1]. The World Health Organization and large epidemiological studies estimate that over onethird of people with diabetes will develop some form of DR during their lifetime [2]. Despite effective treatments such as laser photocoagulation and anti-VEGF therapy, clinical outcomes remain strongly dependent on early detection and diagnosis. Regular screening is therefore essential. However, health systems worldwide face significant challenges in meeting the escalating demand for retinal examinations. In recent years, advances in artificial intelligence and the increasing availability of large-scale retinal imaging datasets have created new opportunities to address these screening gaps at scale.


### A. Clinical Context and Grading

Diagnosis of DR predominantly relies on retinal fundus photography, graded according to the International Clinical Diabetic Retinopathy (ICDR) Severity Scale, which includes five stages from No DR to Proliferative DR [3]. Many screening programs simplify this into a binary classification of referable versus non-referable DR to support operational workflows, though full grading remains important for research and clinical prognosis [4], [5]. Recent reviews, such as Yang et al. (2022) [6], provide a broader historical perspective on DR classification systems, outlining how current ICDR/ETDRS scales capture clinical severity yet overlook neurodegenerative changes, and emerging imaging modalities. Their analysis highlights why ongoing refinement of grading standards is essential in parallel with advances in AI-based screening.

### B. Promise of Deep Learning
Deep learning (DL) rapidly emerged as a powerful approach for DR screening. Gulshan et al. (2016) [7] showed near-expert performance on large private datasets, followed by Krause et al. (2018) [8] and Rakhlin et al. (2018) [9], who extended DL to multi-task grading and related applications. These early successes positioned CNNs as a scalable tool for populationlevel screening. Yet, translation into practice required more than high accuracy; addressing data bottlenecks, integrating into screening programs, and obtaining regulatory approval remained critical hurdles [10].

### C. Challenges in Early Research
Despite their promise, early models also revealed important limitations:

• Reproducibility crisis: Many pioneering studies relied on proprietary datasets and unreleased code, complicating replication and fair benchmarking. This highlighted the urgent need for transparency, open-source implementations, and publicly accessible resources [11].
Generalization gap: Models that performed well in internal validation often showed degraded performance on external datasets due to demographic variability, device differences, and domain shifts. This exposed the necessity of diverse multi-center cohorts and standardized evaluation protocols to ensure reliable deployment [12]. • Task simplification: Early works commonly reduced grading to a binary referable versus non-referable classification, neglecting the richer five-class clinical grading system. While operationally convenient, this limits prognostic insight and clinical adoption beyond initial screening [13], [14]. • Black-box nature: The lack of model interpretability hindered clinical trust and regulatory approval. Without lesion-level explanations or uncertainty quantification, clinicians were cautious about relying solely on automated outputs [15].

D. Objectives and contributions of this survey

The rapid advancement and diverse nature of research into deep learning (DL) methods for diabetic retinopathy (DR) screening have created an urgent need for a comprehensive and systematic synthesis of the field. This survey aims to fulfill this need by providing a critical overview of the evolution, current state, and future trajectory of DL-based approaches to DR screening. The contributions of this work are as follows: 1) Clinical Trust and Reproducibility: We perform a rigorous analysis of reproducibility challenges observed in early DL studies, highlighting recent progress toward open-source implementations, public dataset availability, and standardized evaluation practices that collectively advance clinical reliability. 2) Benchmarking and Datasets: This survey presents a consolidated benchmarking of representative DR screening methods published between 2016 and 2025 (Section VIII) and a curated summary of prominent retinal imaging datasets (Section VII). These resources facilitate contextual understanding of algorithmic performance relative to dataset characteristics and evaluation frameworks. 3) Emerging Research Directions: We discuss emerging methodological trends, such as self-supervised representation learning, federated and domain generalization techniques, and interpretable modeling strategies incorporating clinical expertise. The translational potential of these innovations for large-scale, real-world screening contexts is emphasized. Beyond diabetic retinopathy, many of the reviewed advances,such as self-supervised and federated learning, generalize to other medical imaging domains. Unlike earlier 2022–2023 surveys, this work integrates 2016–2025 studies under one taxonomy linking reproducibility, generalization, and clinical deployment.

The structure of this survey is depicted in Fig. 1, which maps the progression from foundational advances, through identification of key challenges, to subsequent innovations and current efforts aimed at clinical integration.

E. Organization of the paper

The remainder of this survey is organized as follows. Section II introduces foundational CNN-based approaches and outlines the reproducibility crisis. Section III discusses generalization challenges under domain shift, while Section IV reviews methods addressing data scarcity, label noise, and class imbalance. Section V surveys architectural innovations and interpretability techniques, and Section VI covers federated and privacy-preserving training. Section VII and Section VIII consolidate datasets and benchmarking results. Finally, Section X provides a critical discussion, and Section XI concludes with future directions.
    

---

## II. Foundational Breakthroughs and the Reproducibility Gap

Proof of feasibility. The 2016 JAMA study by Gulshan et al. [7] demonstrated that convolutional neural networks (CNNs), trained on a very large internally curated fundus corpus, could detect referable diabetic retinopathy (rDR) at nearexpert performance (AUC ≈ 0.99) on EyePACS and Messidor-2 datasets (refer Section VII; see Fig. 2 for representative fundus examples). This landmark result catalyzed a wave of DL-forDR research and paved the way for clinical systems. Around the same time, Abramoff  et al. (2016) [4] demonstrated high AUC (∼0.98) on Messidor-2 using the proprietary IDx-DR system, highlighting the potential for regulatory-grade screening devices. Concurrent work investigated multi-task prediction (e.g., DR grade, diabetic macular edema, and image quality) and grader variability [8], further normalizing DL as a component of screening programs.

Task formulation and early pipelines. Most early systems framed screening as binary rDR prediction, a clinically aligned triage decision that is easier than fine-grained five-class grading. Kaggle-style pipelines (e.g., Lam et al., 2018; Rakhlin et al., 2018 [5], [9]) experimented with contrast normalization, cropping to the retinal disk, and heavy ensembling (CNN+RNN hybrids). While these competition-driven strategies often reported strong validation accuracy, they rarely included external generalization tests, making it difficult to isolate the key factors contributing to model robustness.

Why numbers were hard to reproduce. Independent reimplementations on public data reported substantially lower performance than originally claimed. Voets et al. [11] showed that reproducing [7] on EyePACS/Messidor-2 datasets with publicly available labels yielded AUCs in the 0.85-0.95 range,depending on label source and split policy. Three recurring factors were identified:
1) Dataset opacity: private curation steps and exclusion criteria were not fully described;

2) Label inconsistency: single-grader labels or differing adjudication rules between datasets introduced systematic shifts;

3) Evaluation leakage: per image rather than per-patient splits inflated metrics when both eyes from a patient appeared across train/test.

Clinical signal vs. dataset artifacts. High internal accuracy sometimes reflected sensitivity to acquisition factors (camera brand, illumination, field-of-view) rather than pathology. Strong quality control and preprocessing (e.g., Contrast Limited Adaptive Histogram Equalization (CLAHE)) improved internal validation yet induced artifacts that failed to translate across sites. Several studies reported sharp drops when re-tested on Messidor or DDR datasets after training on EyePACS/APTOS datasets (refer section VII), highlighting domain shift as the central obstacle to clinical portability [13], [16]. Domain shift remains a core methodological challenge across medical imaging.

Takeaways for the community. The foundational period established (1) DL’s raw capability for fundus screening, and (2) the necessity of transparent data curation, patient-level evaluation, and external testing for credible progress. Early interpretability attempts such as Sayres et al. (2019) [15] also revealed the limitations of post-hoc saliency maps, motivating the field’s later shift toward integrated attention mechanisms. These foundational insights have influenced not only DR research but also the broader development of trustworthy clinical AI methodologies
    
## III. Generalization and Domain Shift

- **Sources of shift**: DR datasets differ along multiple axes: camera hardware (sensor, optics, resolution), capture protocol (mydriasis, field-of-view), demographics (age, ethnicity, comorbidities), prevalence of severe/proliferative diabetic retinopathy (PDR) cases, and label protocol (number of graders, adjudication). A model tuned to the EyePACS distribution can underperform when moved to Messidor-2 or DDR even with identical architectures and training schedules [17]–[19]. Beyond covariate shift, label shift (different grading rubrics) and prior shift (class frequency) further complicate transfer [2], [3]
    
- **Evaluation pitfalls**: Per-image reporting inflates results when the two eyes of a patient are split across folds; classimbalance can allow very high accuracy with poor PDR sensitivity; and strong pre/post-processing may bake in sitespecific artifacts. We recommend (i) per-patient splits and metrics, (ii) explicit thresholds with calibration (ECE/Brier), (iii) confidence intervals, and (iv) at least one external test cohort [8], [11], [15]
What helps in practice. Three families of techniques consistently improve portability:

1) Pretraining and regularization. Large-scale selfsupervised pretraining (contrastive or masked autoencoding) on diverse unlabeled fundus images learns features
less tied to a single site and reduces label requirements
[20]–[23]. Favor simple, robust augmentations (including
color jitter, mild blur, and crop/resize augmentations)
over heavy image manipulation that can embed domainspecific artifacts; GAN-based augmentation should be
validated with an external cohort [24].
2) Domain generalization (DG) and adaptation (DA).
Representation disentanglement and feature normalization (e.g., DECO) explicitly separate disease signal
from domain cues, improving cross-dataset accuracy
without target labels [25]. When unlabeled target data
are available, consistency and pseudo-labeling stabilize
adaptation. GDRNet (Che et al., 2023) [26] and other
DA pipelines [27], [28] showed measurable cross-domain
improvements.
3) Structure and priors. Injecting retinal structure-vessels,
optic disc/macula geometry, and lesion ontologies constrain the hypothesis space. Neuro-symbolic hybrids [29]
and lesion-aware attention discourage shortcuts (camera
edges, illumination rings) and align evidence with clinical
reasoning.
DG/DA methods can still overfit to the collection of source
domains; synthetic style-transfer may improve average accuracy
while harming worst-case groups; and privacy constraints limit
broad data pooling. These motivate federated learning (Sec. VI),
standardized reporting, and multi-center prospective validation.
    
- **Improving Portability**: Recommended methods include self-supervised pretraining , Domain Generalization (DG) like DECO , and injecting retinal structure (vessels, lesions) into neuro-symbolic hybrids.
    

---

## IV. Methodological Advances (2019-2024)

IV. METHODOLOGICAL ADVANCES (2019 - 2024) Following the recognition of core challenges, research from

2019 onward produced a rich set of methodological innovations.

These approaches directly targeted issues of imbalance, data

scarcity, domain shift, and clinical trust.

A. Handling Class Imbalance

Advanced stages of DR (Severe NPDR, PDR) are rare, making datasets highly skewed. Standard CNNs tend to collapse to majority classes, under-detecting sight-threatening disease. To mitigate this:

• Loss functions: Class-balanced focal loss and costsensitive weighting improved minority recall without

excessively harming majority accuracy.

• Synthetic augmentation: GAN-based synthetic image

generation has been used to augment minority classes. Lim

et al. (2020) [24] reported 3 - 5% accuracy improvements

on EyePACS, though concerns remain about unrealistic artifacts. Other augmentation pipelines such as ESRGAN super-resolution and CLAHE preprocessing [16] pushed performance on APTOS, but later work showed such gains often reflected artifacts rather than pathology [30].

• Ensembles: Qin et al. (2022) [31] explored “deep forest”

ensembles, combining multiple classifiers for robustness,

but scalability is limited. Subsequent ensembles with

EfficientNet backbones [32] and self-adaptive stacking

with attention [33] achieved strong performance on Kaggle/EyePACS but introduced challenges for reproducibility

due to architectural complexity.

Despite progress, true resolution requires larger annotated cohorts and careful per-patient balancing.

B. Learning with Limited Labels

Expert grading is costly; datasets like EyePACS include hundreds of thousands of unlabeled images. To leverage these:

• Semi-supervised learning (Semi-SL): Teacher-student

frameworks and pseudo-labeling, e.g., Duan et al. (2022)

[34], boosted performance on APTOS with 50% fewer

labels.

• Self-supervised learning (SSL): Contrastive methods

such as lesion-aware pretraining [22], [23] yielded robust

representations transferable to external cohorts. Arrieta et

al. (2023) [20] showed EyePACS AUC = 0.94 using only

2% labeled data.

• Masked autoencoders (MAE): Vision Transformers

(ViT) pretraining with MAE scales effectively. Yang et

al. (2024) [21] achieved AUC 0.98 with significantly

fewer labels compared to ImageNet-pretrained CNNs.

Related innovations such as equivariant refinement [35]

also emphasized label efficiency and robustness.

C. Domain Generalization and Adaptation

Generalization across datasets remains a core challenge. Approaches range from domain adaptation methods (e.g., pseudo-labeling, adversarial training, and curriculum strategies [27], [28]) to domain generalization techniques such as DECO [25] and GDRNet [26], which disentangle acquisition factors or improve performance on unseen datasets. More recently, neurosymbolic hybrids [29] integrate ViTs with lesion ontologies, achieving notable gains in cross-domain robustness.

D. Building Clinical Trust

Beyond accuracy, clinicians demand interpretability and safety:

• Explainable AI: Post-hoc heatmaps (Grad-CAM, IG) provided lesion-localization cues [15], but can be misleading.

Efforts such as ULBPEZ feature encodings [36] sought

interpretable handcrafted features with mixed success.

• Integrated attention: IDANet (Bhati et al., 2024) [37]

embedded dual attention to highlight pathologic regions,

improving transparency. More recent dual-attention CNNs

[38] extended this idea to handle class imbalance explicitly.

• Uncertainty estimation: Bayesian CNNs and MC dropout

quantified model confidence [39], supporting safe referral

thresholds. Other works incorporated calibration metrics

explicitly during training [12].

Methodological advances shifted the field from proof-ofconcept CNNs to pipelines capable of learning from scarce labels, transferring across domains, and providing clinically relevant explanations. However, robust multi-center validation remains the exception rather than the norm
    

---

## V. Architectures and Interpretability

A. From CNN Backbones to Ensembles
Early DR pipelines relied on CNNs such as Inception-v3
and ResNet-50, pretrained on ImageNet and fine-tuned for
fundus images. Simple preprocessing (e.g., optic disc cropping,
histogram equalization) helped establish strong baselines. To
boost accuracy, ensembles became common: Arora et al. (2024)
[32] showed EfficientNet ensembles achieved ∼86.5% balanced
accuracy on EyePACS with improved calibration. However,
ensembles raise computational costs, limiting scalability and
reproducibility. Fusion-based transfer learning [40] and adaptive
stacking with attention [33] show promise but share these
concerns.
B. Attention Mechanisms
Attention modules emphasize pathological regions while
reducing background noise. Dual-attention CNNs (Bhati et al.,
2024; Hannan et al., 2025) [37], [38] and refinement-based
approaches [41] improve lesion localization, though reliance
on small or single-center datasets limits generalizability.
C. Vision Transformers (ViTs)
Inspired by NLP, ViTs use self-attention over image patches
to capture global context. Huang et al. (2021) [22] showed
lesion-based contrastive pretraining improved EyePACS generalization, while Yang et al. (2024) [21] achieved state-ofthe-art AUC (0.98) with masked autoencoder pretraining using
fewer labels. These models outperform CNNs mainly through
global-context attention and large-scale pretraining [35] , which
improve cross-dataset generalization, though CNNs still excel
on smaller sets due to stronger inductive bias.
D. Neuro-Symbolic and Hybrid Models
Newer models combine neural networks with symbolic priors.
Urooj et al. (2025) [29] integrated ViTs with vascular and
lesion ontologies, boosting cross-domain performance by more
than 5%. Zhou et al. (2025) [42] proposed GPMKLE-Net
with self-paced multi-scale learning, reflecting a trend toward
clinician-aligned, generalizable systems.
E. Interpretability and Clinical Trust
Explainability remains a barrier. Post-hoc methods (e.g.,
Grad-CAM, integrated gradients) can mislead by highlighting
artifacts [15]. Integrating attention directly into models, as
in IDANet, yields more clinically consistent explanations.
Uncertainty estimation (Bayesian CNNs, MC dropout) [39]
supports safe referral, while lightweight CNNs (Gayathri et
al., 2020) [43] show potential for resource-limited settings but
must balance interpretability, efficiency, and accuracy.
Architectural progress in DR AI has moved from conventional CNNs to attention-based, transformer-based, and hybrid
neuro-symbolic models. The field is advancing from reliance
on post-hoc interpretability toward integrated lesion-aware and
uncertainty-driven designs. Nonetheless, multicenter validation
and practical deployment require continued attention to the
trade-offs between accuracy, efficiency, scalability, and clinical
trust.

---

## VI. Federated and Privacy-Preserving Learning

A. Why Federated Learning?

Access to diverse, multi-center retinal datasets is crucial for developing generalizable models in diabetic retinopathy (DR) screening. However, privacy frameworks such as HIPAA and GDPR, as well as institutional policies, severely restrict the transfer of raw patient data across clinical sites. Centralized data pooling is thus often infeasible due to concerns around patient confidentiality, ethics, and logistical barriers. Federated Learning (FL) addresses this challenge by enabling institutions to collaboratively train models without sharing raw images: models are trained locally on institutional data, and only model updates are aggregated on a central server. This decentralized approach enables collaborative learning while maintaining compliance with privacy regulations and data governance requirements.

B. Simulation Studies

Initial DR-focused FL studies generally simulated multicenter scenarios using public datasets. Matta et al. (2023) [44] trained federated CNN ensembles on EyePACS, Messidor, and IDRiD, demonstrating only minor accuracy reductions compared to centralized setups. Mohan Raj et al. (2024) [45] evaluated EfficientNet-based FL across DDR, EyePACS, and APTOS, reporting robust generalization (∼93.2% accuracy) and resilience to lower-quality images. Additional ensemble-based models, including Wong et al. (2023) [46] and Aftab et al. (2025) [40], showcase the potential to adapt fusion and stacking methods for federated or distributed learning, especially for multi-dataset and cross-domain training. Recent innovations such as FedGAN [47] enable privacy-preserving generation of synthetic retinal images, further mitigating data scarcity and regulatory barriers.

C. Challenges and Limitations

Despite these advances, several significant challenges persist:

• Statistical heterogeneity: Non-IID distributions across

institutions (variations in class prevalence, sensors, protocols) can weaken model convergence and performance.

• Communication overhead: Training large models (such

as ViTs or ensembles) exacerbates network bandwidth

usage and synchronization delays relative to centralized

learning.

• Security risks: Even model updates may leak sensitive

information unless augmented by differential privacy,

homomorphic encryption, or secure aggregation protocols.

• Evaluation gaps: Most published FL studies rely on simulations with public datasets, rather than true prospective

clinical deployment and validation.

D. Outlook

Key priorities for future research in DR-FL include:

1) Personalization: Incorporating local adaptation layers

to adjust for site-specific biases and data distributions.

2) Efficiency: Leveraging lightweight models (MobileViT,

quantized CNNs, or efficient CNNs [43]) to minimize

communication and computation costs without sacrificing

accuracy.

3) Security guarantees: Integrating privacy-preserving

cryptography such as differential privacy or secure

aggregation to further protect sensitive information.

4) Prospective validation: Designing and reporting realworld FL deployments that bridge institutions and geographies, assessing robustness and clinical relevance in

true clinical workflows.

Federated learning represents a pivotal advance toward privacy-preserving, multi-center AI in DR screening. While current studies demonstrate feasibility, real-world deployment remains limited; regulatory, communication, and heterogeneity barriers must still be solved before FL achieves sustained multi-hospital validation.


---

## VII. Datasets Overview

VII. DATASETS OVERVIEW

Table I summarizes the most commonly used datasets for DR research, spanning public competitions, curated clinical cohorts, and private collections. In the following subsections, we group them into public competition benchmarks, clinical datasets, and private cohorts to highlight their respective roles and limitations.

A. Public Competition Datasets

The Kaggle EyePACS competition remains the most widely used benchmark. Over 30 of the papers we reviewed report results on EyePACS, though often with differing splits, label protocols, or data cleaning pipelines (e.g., Lam et al. 2018; Rakhlin et al. 2018; Yuan et al. 2020). These inconsistencies complicate fair comparison. APTOS 2019 [48], with 3,662 labeled images, is another widely used benchmark, appearing in at least 15 surveyed works (Farag et al., 2022; Duan et al. 2022; Alwakid et al. 2023; Hannan et al. 2025). However, its small size and imbalance have led to extreme overfitting, with many works reporting suspiciously high accuracies (more than 97%) that fail to generalize.

B. Clinical Datasets

Messidor-2 [18] is the most common external validation set. Dozens of studies (e.g., Gulshan et al. 2016; Voets et al. 2019; Cheung et al. 2021; Che et al. 2023) use Messidor-2 as a hold-out test, often revealing substantial performance

drops compared to Kaggle-split numbers. DDR (Chinese, 13,673 images) [19] has become another external benchmark, featured in Akram et al. 2025; Bhati et al. 2024; Alwakid et al. 2023. IDRiD [49] and FGADR [50] provide detailed lesion annotations, enabling hybrid classification-segmentation models (Li et al. 2019; Zhang et al. 2022). ODIR [51] further broadens the scope by including multi-disease fundus images. Other corpora like DIARETDB0/1, DRIVE, CHASE, STARE, and HRF remain useful for lesion/vessel segmentation tasks and early CNN baselines [55].

C. Private Clinical Cohorts

Several influential papers report training on massive private datasets (Google 2016 [7]; Krause 2018 [8]). While these datasets enable large-scale training, their inaccessibility exacerbates the reproducibility crisis. Replication studies (Voets 2019 [11]; Papadopoulos 2021 [12]) show that results do not always carry over to public sets. Recent federated learning efforts (Matta 2023 [44]; Mohan Raj 2024 [45]) highlight strategies to leverage local private cohorts without centralizing data.

In total, we identified over 20 distinct datasets across 50+ studies. Table I consolidates key properties (size, labels, access). The lack of standardized splits and per-patient protocols remains a barrier: many reported accuracies are not directly comparable, underlining the need for community benchmarks and open science practices.

Table I: Summary of Commonly Used DR Datasets

|**Dataset**|**Year**|**Size**|**Labels/Task**|**Access**|**Notes**|
|---|---|---|---|---|---|
|EyePACS|2015|~88k|5-class; rDR|Public|Most widely used; noisy labels|
|Messidor-2|2014|1,748|5-class; rDR|Public|Common external test set|
|APTOS 2019|2019|3,662|5-class|Public|Severe imbalance; often overfit|
|DDR|2019|13,673|5-class + lesions|Public|Diverse external validation|
|IDRID|2018|516|5-class + masks|Public|Gold standard for lesion masks|
|ODIR|2019|5,000|Multi-disease|Public|Mixed conditions|
|DRIVE|2004|40|Vessel seg.|Public|Small, widely used for vessels|
|mBRSET|2024|5,164|5-class DR|Public|Low-cost mobile camera data|
|Pvt Google|2016|128k|5-class DR|Private|Unreleased; used in|

---

## VIII. Performance Benchmarking
A. Reported Metrics Across Studies
Over the past decade, reported performance on DR benchmarks has ranged from AUC >0.99 in early private datasets
[7] to more modest accuracies (∼85–90%) on public sets with
rigorous splits [11]. Table II consolidates representative results
(2016–2025) and lists a separate line per dataset to avoid
mixing tasks or cohorts. Metrics are reproduced as reported
in each paper. A dash (“–”) indicates the paper did not report
that metric for the given dataset and task.
B. Discussion of Reported Results
Several trends are apparent:
• Inflated early results: AUCs above 0.99 (e.g., Gulshan
et al. 2016) were rarely reproduced on public test sets
[11].
• Public competitions: On APTOS 2019, dozens of papers
report accuracies >90% [31], [34], [56], but generalization
to Messidor/DDR is poor.
• Label efficiency: SSL and MAE pretraining (Arrieta 2023;
Yang 2024; Fan 2024) achieved competitive or superior
performance with fewer labels.
• Trust-oriented models: Attention-based (IDANet [37],
dual-attention CNNs [38]) and Bayesian approaches [39]
emphasize interpretability and safety.
• Cross-domain generalization: DECO [25], GDRNet
[26], and neuro-symbolic hybrids [29] show promise in
mitigating domain shift. 
- Deployment focus: Lightweight CNNs [43] and FL

pipelines [44], [45] highlight practical trade-offs for

clinical use.
Table II: Selected Performance for DR Screening (2016-2025)

|**Paper**|**Dataset**|**Task**|**AUC**|**Acc.**|**Sens.**|**Spec.**|
|---|---|---|---|---|---|---|
|Gulshan (2016) [7]|EyePACS|Bin|0.99|-|0.90|0.98|
|Gulshan (2016) [7]|Messidor-2|Bin|0.99|-|0.87|0.98|
|Abramoff (2016) [4]|Messidor-2|Bin|0.98|-|0.96|0.87|
|Rakhlin (2018) [9]|Messidor-2|Bin|0.97|-|0.99|0.92|
|Voets (2019) [11]|Messidor-2|Bin|0.85|-|0.81|0.68|
|Gayathri (2020) [43]|EyePACS|Bin|-|0.99|1.00|1.00|
|Arrieta (2023) [20]|EyePACS|Bin|0.94|-|-|-|
|Alwakid (2023) [16]|APTOS|5c|-|0.98|0.98|-|
|Yang (2024) [21]|APTOS|Bin|0.98|0.93|0.96|0.95|
|Bhati (2024) [37]|DDR|5c|0.86|0.85|0.82|-|
|Urooj (2025) [29]|Mixed Pub|5c|-|0.63|-|-|
|Zhou (2025) [42]|APTOS+Mess|5c|0.99|0.94|-|-|

---
Paper,Dataset (Pub/Pvt),Task,AUC,Acc.,Kappa,Sens.,Spec.
2016–2018 (Foundational) ,,,,,,,
Gulshan et al. (2016) [7],EyePACS (Pub),Bin,0.99,-,-,0.90,0.98
Gulshan et al. (2016) [7],Messidor-2 (Pub),Bin,0.99,-,-,0.87,0.98
Abramoff et al. (2016) [4],Messidor-2 (Pub),Bin,0.98,-,-,0.96,0.87
Krause et al. (2018) [8],Moorfields (Pvt),Multi,0.98,-,0.84,0.97,0.92
Lam et al. (2018) [5],EyePACS (Pub),Bin,-,0.74,-,-,-
Lam et al. (2018) [5],Messidor (Pub),5c,-,0.57,-,-,-
Rakhlin et al. (2018) [9],EyePACS (Pub),Bin,0.92,-,-,0.92,0.92
Rakhlin et al. (2018) [9],Messidor-2 (Pub),Bin,0.97,-,-,0.99,0.92
2019–2021 (Reproducibility & Early SSL) ,,,,,,,
Voets et al. (2019) [11],EyePACS (Pub),Bin,0.95,-,-,0.90,0.83
Voets et al. (2019) [11],Messidor-2 (Pub),Bin,0.85,-,-,0.81,0.68
Sayres et al. (2019) [15],EyePACS/Messidor (Pub),5c,-,-,-,0.91,0.94
Taufiqurrahman et al. (2020) [14],APTOS 2019 (Pub),5c,-,0.85,0.92,-,-
Lim et al. (2020) [24],EyePACS (Pub),5c,0.98,-,-,-,-
Saxena et al. (2020) [57],Messidor (Pub),Bin,0.95,-,-,0.88,0.90
Saxena et al. (2020) [57],Messidor-2 (Pub),Bin,0.92,-,-,0.81,0.86
Saxena et al. (2020) [57],EyePACS (Pub),Bin,0.92,-,-,0.84,0.89
Gayathri et al. (2020) [43],EyePACS (Pub),Bin,-,0.99,-,1.00,1.00
Gayathri et al. (2020) [43],Messidor (Pub),Bin,-,0.99,-,0.99,0.99
Gayathri et al. (2020) [43],IDRiD (Pub),Bin,-,0.99,-,0.99,0.99
Huang et al. (2021) [22],EyePACS (Pub),5c (SSL),-,-,0.83,-,-
Papadopoulos et al. (2021) [12],IDRiD,Bin,0.86,-,-,-,-
"2022–2023 (Imbalance, SSL, Domain Generalization) ",,,,,,,
Farag et al. (2022) [13],APTOS 2019 (Pub),5c,-,0.82,0.88,-,-
Farag et al. (2022) [13],APTOS 2019 (Pub),Bin,-,0.97,0.94,0.97,0.98
Duan et al. (2022) [34],APTOS 2019 (Pub),5c (Semi-SL),-,0.93,0.91,0.93,-
Che et al. (2022) [28],Messidor (Pub),5c,0.86,0.70,-,-,-
Che et al. (2022) [28],IDRiD (Pub),5c,0.84,0.59,-,-,-
Berbar (2022) [36],EyePACS (Pub),Bin,0.97,0.97,-,1.00,1.00
Berbar (2022) [36],Messidor-2 (Pub),3c,-,0.98,-,1.00,1.00
Berbar (2022) [36],EyePACS (Pub),3c,-,0.96,-,1.00,0.96
Qin et al. (2023) [31],EyePACS (Pub),5c,-,0.74,-,0.74,-
Arrieta et al. (2023) [20],EyePACS (Pub),Bin,0.94,-,-,-,-
Arrieta et al. (2023) [20],Messidor-2 (Pub),Bin,0.89,-,-,-,-
Alwakid et al. (2023) [16],APTOS 2019 (Pub),5c,-,0.98,-,0.98,-
Alam et al. (2023) [23],"UIC private (Pvt, external)",Bin,0.91,-,-,-,-
Mohanty et al. (2023) [56],APTOS 2019 (Pub),5c,-,0.97,-,-,-
Wong et al. (2023) [46],APTOS 2019 (Pub),5c,-,0.82,-,-,-
Wong et al. (2023) [46],EyePACS+Messidor (Pub),3c,-,0.75,-,-,-
Bodapati & Balaji (2023) [33],APTOS 2019 (Pub),5c,0.96,0.86,0.89,0.72,-
Matta et al. (2023) [44],OPHDIAT,5c (FL),0.95,-,-,-,-
"2024–2025 (Transformers, Federated, Hybrids) ",,,,,,,
Arora et al. (2024) [32],EyePACS (Pub),5c,-,0.86,-,-,-
Shakibania et al. (2024) [58],APTOS 2019 (Pub),5c,0.89,0.83,0.89,0.83,0.94
Shakibania et al. (2024) [58],APTOS 2019 (Pub),Bin,0.97,0.97,0.95,0.98,0.97
Bhati et al. (2024) [37],DDR (Pub),5c,0.86,0.85,-,0.82,-
Bhati et al. (2024) [37],EyePACS (Pub),5c,0.96,0.93,-,0.92,-
Bhati et al. (2024) [37],IDRiD (Pub),5c,0.93,0.90,-,0.84,-
Yang et al. (2024) [21],APTOS 2019 (Pub),Bin,0.98,0.93,-,0.96,0.95
Xia et al. (2024) [25],FGADR (Pub),5c,0.86,0.57,-,-,-
Fan et al. (2024) [35],EyePACS (Pub),5c (SSL),0.93,0.87,0.85,-,-
Mohan Raj et al. (2024) [45],EyePACS (Pub),5c (FL),-,0.90,-,-,-
Urooj et al. (2025) [29],EyePACS / APTOS / Messidor (Pub),5c (Hybrid),-,0.63,-,-,-
Wang et al. (2025) [41],APTOS 2019 (Pub),5c,0.89,-,0.82,0.80,0.81
Zhou et al. (2025) [42],APTOS + Messidor (Pub),5c,0.99,0.94,-,-,-
Aftab et al. (2025) [40],APTOS + IDRiD + Messidor-2 (Pub),Bin / 5c,-,0.96,-,-,-
Hannan et al. (2025) [38],APTOS 2019 (Pub),5c,-,0.83,-,-,-
Akram et al. (2025) [39],APTOS + DDR (Pub),5c,-,0.97,-,0.97,-
Ahmed & Bhuiyan (2025) [59],APTOS 2019 (Pub),Bin,0.99,0.98,-,0.99,-
Ahmed & Bhuiyan (2025) [59],APTOS 2019 (Pub),5c,0.94,0.84,-,0.63,-

## IX. Evaluation Protocols and Reporting Standards

Why protocols matter. Across the 50+ papers we reviewed, reported performance varies widely, often due to inconsistent splits, label protocols, and thresholds. Without transparent reporting, comparisons devolve into apples-to-oranges.

A. Per-Patient vs. Per-Image

Because two eyes from a single patient are statistically dependent, splitting the left and right eyes across train/test inflates accuracy. We recommend reporting both per-patient and per-image metrics, with per-patient as primary. This reduces optimistic bias and aligns with clinical decision-making (patients, not images, are referred). Papadopoulos et al. (2021) [12] explicitly highlighted the gap between patient- and imagelevel AUCs, while competition-style studies on APTOS (e.g., Alwakid et al., 2023 [16]) may have inadvertently benefited from per-image leakage.

B. External Validation

Models tuned to EyePACS frequently degrade on Messidor- 1/2 or DDR due to domain shift. At least one external test set should be included (e.g., train on EyePACS, test on Messidor-2 [7], [11] or DDR [19]). When possible, include more than one external site to avoid overfitting to a single target distribution. For example, Alwakid et al. (2023) [16] reported near-perfect accuracy on APTOS but only ∼80% on DDR, while Che et al. (2023) [26] demonstrated how multi-dataset training can mask poor external generalization. Authors should also specify dataset versions or release identifiers, since EyePACS and other corpora exist in multiple variants whose differences significantly affect reported performance.

C. Thresholds, Calibration, and Confidence Intervals

Binary rDR detection depends on threshold choice; fiveclass grading depends on confusion structure. Authors should report ROC/PR curves, operating points, and calibration metrics (e.g., ECE, Brier score). Confidence intervals (95% CIs) via bootstrapping at the patient level and statistical tests (e.g., McNemar’s) are essential. Calibration supports safe referral threshold setting. Bayesian models [39] and ensembles [33] illustrate uncertainty quantification benefits.

D. Label Quality and Adjudication

Clear documentation of label sourcing (single vs. multiple graders, adjudication, rubric) is critical. Label protocol variability can cause large AUC swings [8], [11]. Soft labels or adjudicated subsets help evaluate robustness. Semi-supervised and data-fusion approaches [20], [40] highlight risks from label scarcity and oversampling.

E. Preprocessing and Leakage Checks

All preprocessing steps (cropping, CLAHE, GAN augmentation, super-resolution) must be documented, with patient-level splits verified to prevent leakage. Site-specific artifact induction can undermine generalization [11]. Notably, GAN/ESRGANbased gains reported in [16], [24], [30] often reflect synthetic artifacts rather than true pathology. Released code should include deterministic seeds, explicit split indices, and computational details.

---

## X. Discussion and Future Directions

Where we are. The field has moved from proof-of-concept CNNs on private datasets [7], [8] to approaches tackling domain shift [25], [26], [29], label scarcity [20]–[23], and clinical trust [15], [37], [38]. Yet three key gaps remain.

A. Robust generalization at scale

Most work relies on single-site data or simulated mixes. Promising results, such as disentanglement [25] or neurosymbolic priors [29], still lack broad validation. Drops like APTOS ∼98% vs. DDR ∼80% [16] stress the need for prospective, multi-center studies with harmonized metadata and pre-registered protocols.

B. Clinically useful five-class grading

Binary rDR triage is common, but five-class grading better supports care. High reported accuracies on APTOS [13], [16], [56] may reflect imbalance or leakage. Next steps include stable per-class sensitivity (especially Severe NPDR/PDR), uncertainty-aware referral [39], and patient-level outcomes such as time-to-referral, vision loss avoided, and cost-effectiveness.

C. Reproducibility and transparency

Topline metrics are not enough. Norms should include code/model release, split files, and dataset cards. For private cohorts, publish acquisition details and share synthetic audit sets. Replication efforts [11], [12] highlight reproducibility as a core objective.

D. Promising directions

Label-efficient pretraining: Fundus-tailored SSL reduces labeling costs [20]–[22], [35]. Domain generalization & priors: Structural and lesion priors mitigate shortcut learning [25], [26], [29]. Federated learning: Move beyond simulations to real-world deployments with secure aggregation, personalization, and compliance [44], [45]. Trust by design: Prefer integrated lesion/attention modules [37], [38], [41] with calibrated uncertainty and human-in-the-loop review [15], [33]. Efficiency and deployment: Lightweight/quantized models [40], [43] are vital for use in low-resource settings.

E. Agenda for 2025 onwards

Priorities include: (i) open multi-center benchmarks (EyePACS, Messidor-2, DDR, FGADR/ODIR) with fixed perpatient splits; (ii) broader reporting with calibration metrics and “operating point sheets” detailing thresholds, predictive values, and costs; (iii) lightweight “results cards” documenting dataset versions, protocols, and confidence intervals; and (iv) stronger code/model sharing and replication. Finally, deploying federated and privacy-preserving models with differential privacy and clear governance is key to responsible crossinstitutional use.


## XI. Conclusion

Deep learning for DR has advanced toward addressing imbalance, scarcity, and trust. Clinical adoption hinges on calibrated per-patient evaluation and rigorous multi-center validation. By embracing open science and embedding retinal priors, the field can move from pixels to patient benefit.Deep learning for DR has advanced from private CNN studies [7], [8] toward systems addressing imbalance [13], [16], label scarcity [20], [34], domain shift [25], [26], [29], and trust via attention, uncertainty, and hybrid models [15], [37], [38]. Federated learning [44], [45] and lightweight architectures [40], [43] push deployment into resource-constrained clinics.

Clinical adoption hinges on reproducibility, calibrated perpatient evaluation, and rigorous multi-center validation. Standardized protocols, dataset cards, and open benchmarks remain

prerequisites. By embracing open science, embedding retinal priors, and validating across diverse cohorts, the field can move from pixels to patient benefit at scale.

---

# Reference
REFERENCES
[1] International Diabetes Federation, IDF Diabetes Atlas, 10th Edition,
2021. Brussels, Belgium: IDF, 2021, global estimates and projections.
[2] R. Lee et al., “Epidemiology of diabetic retinopathy, diabetic macular
edema and related vision-threatening complications,” Survey of Ophthalmology, vol. 60, no. 4, pp. 311–337, 2015.
[3] C. P. Wilkinson, F. L. Ferris, R. E. Klein, P. P. Lee, C. D. Agardh, S. Davis,
D. Dills, M. Kampik, D. Pararajasegaram, and J. Porta, “Proposed
international clinical diabetic retinopathy and diabetic macular edema
disease severity scales,” Ophthalmology, vol. 110, no. 9, pp. 1677–1682,
2003.
[4] M. D. Abramoff ` et al., “Improved automated detection of diabetic
retinopathy on a publicly available dataset,” Investigative Ophthalmology
& Visual Science, 2016.
[5] C. Lam, D. Yi, M. Guo, and T. Lindsey, “Automated detection of diabetic
retinopathy using deep learning,” in AMIA Joint Summits on Translational
Science Proceedings. American Medical Informatics Association, 2018,
pp. 147–155.
[6] Z. Yang, T.-E. Tan, Y. Shao, T. Y. Wong, and X. Li, “Classification of
diabetic retinopathy: Past, present and future,” Frontiers in Endocrinology,
vol. 13, p. 1079217, 12 2022.
[7] V. Gulshan, L. Peng, M. Coram et al., “Development and validation of
a deep learning algorithm for detection of diabetic retinopathy in retinal
fundus photographs,” JAMA, 2016.
[8] J. Krause, V. Gulshan, E. Rahimy, P. Karth, K. Widner, G. Corrado,
L. Peng, and D. Webster, “Grader variability and the importance of
reference standards for evaluating machine learning models for diabetic
retinopathy,” Ophthalmology, vol. 125, 10 2018.
[9] A. Rakhlin, “Diabetic retinopathy detection through integration of deep
learning classification framework,” 06 2018.
[10] Google AI Blog, “Helping doctors detect diabetic retinopathy using
artificial intelligence,” https://blog.google/around-the-globe/google-asia/
arda-diabetic-retinopathy-india-thailand/, 2016, accessed: 2025-09-15.
[11] M. Voets, K. Møllersen, and L. A. Bongo, “Reproduction study
of google’s diabetic retinopathy detection algorithm,” arXiv preprint
arXiv:1803.04337, 2019. [Online]. Available: https://arxiv.org/abs/1803.
04337
[12] A. Papadopoulos, F. Topouzis, and A. Delopoulos, “An interpretable
multiple-instance approach for the detection of referable diabetic retinopathy in fundus images,” Scientific Reports, vol. 11, 07 2021.
[13] M. Farag, M. Fouad, and A. Abdel-Hamid, “Automatic severity classification of diabetic retinopathy based on densenet and convolutional block
attention module,” IEEE Access, vol. 10, pp. 1–1, 01 2022.
[14] S. Taufiqurrahman, A. Handayani, B. R. Hermanto, and T. L. E. R.
Mengko, “Diabetic retinopathy classification using a hybrid and efficient
mobilenetv2-svm model,” in 2020 IEEE REGION 10 CONFERENCE
(TENCON), 2020, pp. 235–240.
[15] R. Sayres, A. Taly, E. Rahimy, K. Blumer, D. Coz, N. Hammel, J. Krause,
A. Narayanaswamy, Z. Rastegar, D. Wu, S. Xu, S. Barb, A. Joseph,
M. Shumski, J. Smith, A. Sood, G. Corrado, L. Peng, and D. Webster,
“Using a deep learning algorithm and integrated gradients explanation
to assist grading for diabetic retinopathy,” Ophthalmology, vol. 126, 12
2019.
[16] G. Alwakid, W. Gouda, and M. Humayun, “Enhancement of diabetic
retinopathy prognostication using deep learning, clahe, and esrgan,”
Diagnostics, vol. 13, no. 14, p. 2375, 2023.
[17] “Kaggle eyepacs diabetic retinopathy detection,” https://www.kaggle.com/
c/diabetic-retinopathy-detection, 2015.
[18] “Messidor-2,” http://www.adcis.net/en/third-party/messidor2/, 2014.
[19] T. Li, Y. Gao, K. Wang, S. Guo, H. Liu, and H. Kang, “Diagnostic
assessment of deep learning algorithms for diabetic retinopathy screening,”
Information Sciences, vol. 501, pp. 511 – 522, 2019. [Online]. Available:
http://www.sciencedirect.com/science/article/pii/S0020025519305377
[20] J. M. A. Ramos, O. J. Perdomo, and F. A. Gonzalez, “Deep semi- ´
supervised and self-supervised learning for diabetic retinopathy detection,”
Proceedings of SPIE (Medical Imaging: Image Processing Applications),
2023, also available as arXiv preprint.
[21] S. Yang et al., “Vision transformers with masked autoencoder pretraining
for referable diabetic retinopathy,” PLOS ONE, 2024.[22] Y. Huang, L. Lin, P. Cheng, J. Lyu, and X. Tang, Lesion-Based
Contrastive Learning for Diabetic Retinopathy Grading from Fundus
Images. springer, 09 2021, pp. 113–123.
[23] M. N. Alam, R. Yamashita, V. Ramesh, T. Prabhune, J. I. Lim,
R. V. P. Chan, J. Hallak, T. Leng, and D. Rubin, “Contrastive learningbased pretraining improves representation and transferability of diabetic
retinopathy classification models,” Scientific Reports, vol. 13, no. 1, p.
6047, 2023.
[24] G. Lim, P. Thombre, M. L. Lee, and W. Hsu, “Generative data
augmentation for diabetic retinopathy classification,” in 2020 IEEE 32nd
international conference on tools with artificial intelligence (ICTAI).
IEEE, 2020, pp. 1096–1103.
[25] P. Xia et al., “Generalizing to unseen domains in diabetic retinopathy with
disentangled representations (deco),” in MICCAI, 2024, code available.
[26] H. Che, Y. Cheng, H. Jin, and H. Chen, Towards Generalizable Diabetic
Retinopathy Grading in Unseen Domains. Springer, 10 2023, pp. 430–
440.
[27] G. Zhang, B. Sun, Z. Zhang, J. Pan, W. Yang, and Y. Liu, “Multi-model
domain adaptation for diabetic retinopathy classification,” Frontiers in
Physiology, vol. 13, p. 918929, 07 2022.
[28] H. Che, H. Jin, and H. Chen, “Learning robust representation for
joint grading of ophthalmic diseases via adaptive curriculum and
feature disentanglement,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention. Springer, 2022, pp.
523–533.
[29] M. Urooj, A. Banerjee, F. Shaikh, K. Thakur, and S. Gupta, “Single
domain generalization in diabetic retinopathy: A neuro-symbolic learning
approach,” 09 2025.
[30] G. Alwakid, W. Gouda, M. Humayun, and N. Jhanjhi, “Enhancing diabetic
retinopathy classification using deep learning,” DIGITAL HEALTH, vol. 9,
09 2023.
[31] X. Qin, D. Chen, Y. Zhan, and D. Yin, “Classification of diabetic
retinopathy based on improved deep forest model,” Biomedical signal
processing and control, vol. 79, p. 104020, 2023.
[32] L. Arora, S. Singh, S. Kumar, H. Gupta, W. Alhalabi, V. Arya, S. Bansal,
K. Chui, and B. B. Gupta, “Ensemble deep learning and efficientnet for
accurate diagnosis of diabetic retinopathy,” Scientific Reports, vol. 14,
12 2024.
[33] J. Bodapati and B. Balaji, “Self-adaptive stacking ensemble approach
with attention based deep neural network models for diabetic retinopathy
severity prediction,” Multimedia Tools and Applications, vol. 83, pp.
1–20, 05 2023.
[34] S. Duan, P. Huang, M. Chen, T. Wang, X. Sun, M. Chen, X. Dong,
Z. Jiang, and D. Li, “Semi-supervised classification of fundus images
combined with cnn and gcn,” Journal of Applied Clinical Medical Physics,
vol. 23, 08 2022.
[35] J. Fan, T. Yang, H. Wang, H. Zhang, W. Zhang, M. Ji, and J. Miao, “A
self-supervised equivariant refinement classification network for diabetic
retinopathy classification,” Journal of Imaging Informatics in Medicine,
vol. 38, 09 2024.
[36] M. Berbar, “Features extraction using encoded local binary pattern for
detection and grading diabetic retinopathy,” Health Information Science
and Systems, vol. 10, 06 2022.
[37] A. Bhati, N. Gour, P. Khanna, A. Ojha, and N. Werghi, “Idanet: An
interpretable dual attention network for diabetic retinopathy grading,”
Artificial Intelligence in Medicine, 2024.
[38] A. Hannan, Z. Mahmood, R. Qureshi, and H. Ali, “Enhancing diabetic
retinopathy classification accuracy through dual-attention mechanism
in deep learning,” Computer Methods in Biomechanics and Biomedical
Engineering: Imaging & Visualization, vol. 13, no. 1, p. 2539079, 2025.
[39] M. Akram, M. Adnan, S. F. Ali, J. Ahmad, A. Yousef, T. A. N. Alshalali,
and Z. A. Shaikh, “Uncertainty-aware diabetic retinopathy detection
using deep learning enhanced by bayesian approaches,” Scientific Reports,
vol. 15, no. 1, p. 1342, 2025.
[40] S. Aftab and S. Akhtar, “Diabetic retinopathy severity classification
using data fusion and ensemble transfer learning,” Journal of Software
Engineering and Applications, vol. 18, pp. 1–23, 01 2025.
[41] Z. Wang, Y. Wang, C. Ma, X. Bao, and Y. Li, “Diabetic retinopathy
classification using a multi-attention residual refinement architecture,”
Scientific Reports, vol. 15, no. 1, p. 29266, 2025.
[42] Q. Zhou, Y. Guo, W. Liu, Y. Liu, and Y. Lin, “Enhancing pathological
feature discrimination in diabetic retinopathy multi-classification with
self-paced progressive multi-scale training,” Scientific Reports, vol. 15,
07 2025.
[43] G. S., V. Gopi, and P. Ponnusamy, “A lightweight cnn for diabetic
retinopathy classification from fundus images,” Biomedical Signal
Processing and Control, vol. 62, p. 102115, 09 2020.
[44] S. Matta, M. Hassine, C. Lecat, L. Borderie, A. Guilcher, P. Massin,
B. Cochener, M. Lamard, and G. Quellec, “Federated learning for diabetic
retinopathy detection in a multi-center fundus screening network,” in
Proceedings of the 45th Annual International Conference of the IEEE
Engineering in Medicine & Biology Society (EMBC), vol. 2023, 07 2023,
pp. 1–4.
[45] G. Raj, M. Morley, and M. Eslami, “Federated learning for diabetic
retinopathy diagnosis: Enhancing accuracy and generalizability in underresourced regions,” 10 2024.
[46] W. K., F. Juwono, and C. Capriono, “Diabetic retinopathy detection
and grading: A transfer learning approach using simultaneous parameter
optimization and feature-weighted ecoc ensemble,” IEEE Access, vol. PP,
pp. 1–1, 01 2023.
[47] H. Kamran, S. Hussain, S. Latif, I. Soomro, M. Alnfiai, and N. Alotaibi,
“Fedgan: Federated diabetic retinopathy image generation,” PLOS One,
vol. 20, 07 2025.
[48] “Aptos 2019 blindness detection (kaggle),” https://www.kaggle.com/
competitions/aptos2019-blindness-detection, 2019.
[49] P. Porwal, S. Pachade, R. Kamble, M. Kokare, G. Deshmukh,
V. Sahasrabuddhe, and F. Meriaudeau, “Indian diabetic retinopathy
image dataset (idrid),” 2018. [Online]. Available: https://dx.doi.org/10.
21227/H25W98
[50] Y. Zhou, B. Wang, L. Huang, S. Cui, and L. Shao, “A benchmark for
studying diabetic retinopathy: Segmentation, grading, and transferability,”
IEEE Transactions on Medical Imaging, vol. 40, no. 3, pp. 818–828,
2021.
[51] “Ocular disease intelligent recognition (odir),” https://odir2019.grandchallenge.org/, 2019.
[52] M. Alam, R. Yamashita, V. Ramesh, T. Prabhune, J. Lim, R. Chan,
J. Hallak, T. Leng, and D. Rubin, “Contrastive learning-based pretraining
improves representation and transferability of diabetic retinopathy
classification models,” 08 2022.
[53] C. Wu, D. Restrepo, L. Nakayama, L. Ribeiro, Z. Shuai, N. Barboza,
M. Sousa, R. Fitterman, A. Pereira, C. Regatieri, J. Stuchi, F. Malerbi,
and R. Andrade, “mbrset: A portable retina fundus photos benchmark
dataset for clinical and demographic prediction,” 07 2024.
[54] J. Mao, X. Ma, Y. Bi, and R. Zhang, “Tjdr: A high-quality
diabetic retinopathy pixel-level annotation dataset,” arXiv preprint
arXiv:2312.15389, 2023.
[55] U. Bhimavarapu and G. Battineni, “Deep learning for the detection
and classification of diabetic retinopathy with an improved activation
function,” in Healthcare, vol. 11, no. 1. MDPI, 2022, p. 97.
[56] C. Mohanty, S. Mahapatra, B. Acharya, F. Kokkoras, V. C. Gerogiannis,
I. Karamitsos, and A. Kanavos, “Using deep learning architectures for
detection and classification of diabetic retinopathy,” Sensors, vol. 23,
no. 12, p. 5726, 2023.
[57] G. Saxena, D. K. Verma, A. Paraye, A. Rajan, and A. Rawat, “Improved
and robust deep learning agent for preliminary detection of diabetic
retinopathy using public datasets,” Intelligence-Based Medicine, vol. 3,
p. 100022, 2020.
[58] H. Shakibania, S. Raoufi, B. Pourafkham, H. Khotanlou, and M. Mansoorizadeh, “Dual branch deep learning network for detection and stage
grading of diabetic retinopathy,” Biomedical Signal Processing and
Control, vol. 93, p. 106168, 2024.
[59] F. Ahmed and M. A. N. Bhuiyan, “Robust five-class and binary diabetic
retinopathy classification using transfer learning and data augmentation,”
07 2025.