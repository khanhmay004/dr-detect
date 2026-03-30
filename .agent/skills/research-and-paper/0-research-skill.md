# research-skill.md

> **Scope**: Researching, retrieving, summarising, and synthesising academic papers on Deep Learning for Diabetic Retinopathy (DR) severity grading.
> **Load when**: The user asks to find, read, review, summarise, or map out papers related to DR screening, grading, or any associated methodology.

---

## Purpose

This skill governs how to handle academic papers in the context of DR severity grading research. It covers:

- Retrieving papers from web sources or local files
- Producing structured, consistent paper summaries
- Synthesising across multiple papers (comparison tables, literature maps, gap identification)
- Maintaining citation discipline — no claim without a source

---

## 1. Source Handling

Papers arrive in three forms. Handle each as follows:

### 1a. User pastes a link or arXiv / PubMed ID
Use `web_fetch` to retrieve the abstract and, where available, the full paper. For arXiv, always fetch the abstract page first (`https://arxiv.org/abs/<id>`), then the PDF if deeper analysis is needed (`https://arxiv.org/pdf/<id>`). For PubMed, fetch the abstract at `https://pubmed.ncbi.nlm.nih.gov/<pmid>/`.

### 1b. User uploads a PDF
Read the PDF/Markdown file in full. Extract structured content in the order: title, authors, venue/year, abstract, methodology, experiments, results, limitations, references.

### 1c. User asks to find papers on a topic
Run at least two distinct web searches with different query angles before concluding. Example queries for the same topic:

```
"diabetic retinopathy grading deep learning transformer 2023"
"DR severity classification fundus CNN benchmark APTOS Messidor"
```

After retrieval, fetch the full abstract or landing page for each candidate. Do not summarise from search snippets alone — snippet text is insufficient for accurate attribution.

---

## 2. Paper Intake Protocol

Before summarising any paper, extract and verify these fields. If a field cannot be determined from the paper, mark it explicitly as `Not reported`:

```
Title       :
Authors     :
Venue       : [journal or conference name + year]
Task        : [e.g., 5-class grading, binary referable/non-referable]
Datasets    : [names + sizes if stated]
Backbone    : [e.g., ResNet-50, EfficientNet-B4, ViT-B/16]
Loss fn     : [e.g., cross-entropy, ordinal regression, MSE on labels]
Preprocessing: [e.g., CLAHE, green channel extraction, BEN normalisation]
Augmentation: [e.g., random flip, colour jitter, Mixup]
Metrics     : [AUC, QWK, sensitivity, specificity, accuracy — note which]
Key results : [headline numbers on primary benchmark]
Code/data   : [URL or "Not released"]
```

Never infer values not stated in the paper. If a backbone is not named, write `Not reported`, not a guess.

---

## 3. Structured Summary Template

After intake, produce the summary in this exact structure:

---

### [Paper Title] — [First Author et al., Year]

**Problem framing**
One paragraph. What clinical or technical problem does this paper address? State the grading scale used (binary, 4-class, 5-class per ICDR). Note whether the target is a screening task (sensitivity-optimised) or a clinical grading task (agreement with ophthalmologists).

**Methodology**
Describe the pipeline in order:
1. Input representation — image modality (colour fundus, green channel, etc.), resolution, preprocessing
2. Architecture — backbone, any modifications, attention mechanisms, multi-scale design
3. Training objective — loss function and rationale given in the paper
4. Any domain-specific design choices (e.g., curriculum learning for label noise, ordinal constraints)

Where the paper references a specific equation or algorithm, note the equation number.

**Key contributions**
Bullet list. State only what the authors themselves claim as novel. Do not editorially inflate contributions.

**Experimental results**
Table if multiple datasets or baselines are compared. Always include the metric name alongside the number — a bare `0.97` means nothing without knowing if it is AUC or accuracy.

| Dataset | Metric | This paper | Best reported baseline |
|---|---|---|---|
| | | | |

**Limitations**
State limitations the authors acknowledge. Then add any limitations not acknowledged but apparent from the methodology or experimental design. Label the latter clearly as `[Reviewer note]`.

**Reproducibility**
- Code: [URL or Not released]
- Pre-trained weights: [URL or Not released]
- Dataset: [public / gated / private]
- Notes: [e.g., training details incomplete, custom split not described]

---

## 4. Multi-Paper Synthesis

When the user asks to compare, map, or review multiple papers, apply the following.

### 4a. Comparison table

Always anchor the table on a shared task and shared metrics. If papers use different metrics, include both and note the discrepancy — do not convert or approximate numbers across metric spaces.

Minimum columns: `Paper | Backbone | Dataset | Primary Metric | Score | Code`

Add columns only when they are populated for at least half the papers in the table. Sparse columns obscure more than they reveal.

### 4b. Methodology taxonomy

Group papers along meaningful axes for DR grading research:

| Axis | Categories |
|---|---|
| Architecture family | CNN (ResNet/EfficientNet/DenseNet), ViT, Hybrid CNN-Transformer, MIL |
| Training objective | Standard CE, Ordinal regression, Label distribution learning, Self-supervised pre-training |
| Label handling | Clean labels, Noisy label learning, Ensemble grading |
| Data regime | Fully supervised, Semi-supervised, Few-shot |
| Preprocessing | BEN normalisation, Green channel, CLAHE, Fundus ROI cropping |

Do not assign a paper to a category unless the paper explicitly describes that approach.

### 4c. Literature gap identification

After mapping the taxonomy, identify gaps as intersections of axes with no or few papers. Frame each gap as a falsifiable research question, not a vague suggestion.

**Example gap format:**
> *Gap*: No published work applies ordinal regression losses to ViT-based architectures on Messidor-2.
> *Why it matters*: ViTs trained with standard CE ignore the ordinal structure of ICDR grades; ordinal losses have improved QWK by X points in CNN settings (cite). Whether this transfers to ViTs is unknown.

---

## 5. DR Domain Reference

Use these facts as anchors when contextualising papers. Do not present them as your own analysis — treat them as a shared reference frame.

### Grading scales

**International Clinical DR (ICDR) scale** (the standard in most DL papers):

| Grade | Label | Clinical meaning |
|---|---|---|
| 0 | No DR | No abnormalities |
| 1 | Mild NPDR | Microaneurysms only |
| 2 | Moderate NPDR | More than mild, less than severe |
| 3 | Severe NPDR | 4-2-1 rule haemorrhages/IRMA |
| 4 | Proliferative DR | Neovascularisation / vitreous haemorrhage |

**Referable DR threshold** (binary task used in screening): grades ≥ 2 are typically considered referable. Some works use ≥ 1. Note which threshold a paper uses — it affects sensitivity/specificity comparisons.

### Benchmark datasets (canonical references)

| Dataset | Size | Labels | Notes |
|---|---|---|---|
| Messidor-2 | 1,748 images | 5-class ICDR + adjudicated | Standard for AUC reporting; Abràmoff et al. 2016 used this |
| APTOS 2019 | 3,662 images | 5-class ICDR | Kaggle competition; label noise known |
| EyePACS | ~88,000 images | 5-class | Large scale; noisy crowd-sourced labels |
| IDRiD | 516 images | 5-class + lesion masks | Used for lesion-level work; smaller scale |

When a paper reports results only on APTOS 2019 without Messidor-2, flag this: APTOS labels are known to be noisier and results are harder to compare across papers.

### Standard metrics for DR grading

| Metric | Abbreviation | Used for |
|---|---|---|
| Area under ROC curve | AUC | Binary / one-vs-rest grading |
| Quadratic weighted kappa | QWK | Ordinal agreement with ground truth |
| Sensitivity (recall) | Se | Clinical screening (prioritise) |
| Specificity | Sp | Clinical screening (report alongside Se) |
| Accuracy | Acc | Multi-class; less informative alone due to class imbalance |

When a paper reports only accuracy on an imbalanced dataset without sensitivity/specificity or AUC, note this as a limitation.

### Preprocessing vocabulary

- **BEN (Ben Graham normalisation)**: Subtract local average colour, normalise luminance. Introduced by Graham (2015) Kaggle winner; widely used since.
- **Green channel extraction**: Green channel of RGB fundus has highest contrast for retinal structures.
- **CLAHE**: Contrast-limited adaptive histogram equalisation; improves microaneurysm visibility.
- **Fundus ROI cropping**: Remove black border around fundus image before training.

---

## 6. Citation Discipline

Every factual claim in a summary or synthesis must be traceable to a specific source:

- If from the paper being summarised: implicit (no additional citation needed)
- If from another paper: cite author + year inline, e.g., `(Gulshan et al., 2016)`
- If from a dataset description or benchmark: cite the dataset paper
- If uncertain: write `[Source unclear — verify]` and do not state the claim as fact

Never smooth over contradictions between papers. If two papers report conflicting results on the same benchmark, state both results, note the discrepancy, and identify likely reasons (different splits, different preprocessing, different referable thresholds).

---

## 7. Output Format Summary

| Request type | Output |
|---|---|
| Single paper summary | Full structured template (Section 3) |
| Compare N papers | Intake table + comparison table + methodology taxonomy |
| Find papers on topic | List of candidates with intake fields + structured summaries |
| Literature review / gap analysis | Taxonomy (Section 4b) + gap list (Section 4c) + comparison table |
| "What do we know about X" | Synthesis paragraph with inline citations + supporting table |