# evaluate.md

> **Scope**: Critically evaluating ideas, hypotheses, and research directions in the context of deep learning for DR severity grading.
> **Load when**: The user proposes an architecture, a training strategy, a loss function, or a research direction and wants rigorous assessment — not encouragement.

---

## Purpose

This skill governs how to evaluate ideas against existing evidence in DR research. The goal is not to validate ideas but to pressure-test them: find what is supported, what is uncertain, and what is contradicted by the literature.

**Core operating principle**: Every evaluative claim is either:
- **Evidence-based**: directly supported by a cited paper, dataset analysis, or documented benchmark result
- **Inference**: reasoned from established principles, clearly labelled as `[Inference]`
- **Unknown**: no sufficient basis exists; state this explicitly and stop, or reason under uncertainty with full transparency

Do not blend these categories. Do not present inference as evidence.

---

## 1. Input Types and Handling

### Type A — Architecture idea

The user proposes a new model design, modification, or component (e.g., "add a cross-attention module between scales", "replace the classification head with an ordinal regression layer").

**Evaluation protocol:**

1. **Prior art search**: Search for existing work that uses the same or analogous component in DR or related medical imaging tasks. Use `0-research-skill.md` protocols for retrieval.
2. **Mechanism analysis**: Explain what the proposed component does computationally and why it might (or might not) address a real limitation of current DR grading models.
3. **Evidence for the mechanism**: Cite papers where this mechanism has been tested. Note whether those papers are in-domain (fundus / medical imaging) or out-of-domain (natural images). Out-of-domain evidence is weaker — label it accordingly.
4. **Potential failure modes**: Identify concrete reasons the idea might underperform. Ground each failure mode in a known phenomenon (e.g., label noise in APTOS 2019, class imbalance in all DR datasets, limited dataset size relative to model capacity).
5. **Verdict**: Summarise as one of:
   - `Supported`: meaningful evidence exists that this direction works
   - `Plausible`: mechanism is sound, limited direct evidence
   - `Contested`: conflicting evidence exists — describe the conflict
   - `Unsupported`: no evidence; identify what experiment would test it

### Type B — Training strategy or loss function

The user proposes a specific way to train a DR grading model (e.g., "use focal loss for class imbalance", "apply label smoothing", "use curriculum learning sorted by grade confidence").

**Evaluation protocol:**

1. **Problem it addresses**: State precisely what limitation in standard DR training this strategy targets (class imbalance, label noise, ordinal structure, etc.). If the user has not stated this, ask before evaluating.
2. **Literature check**: Search for papers applying this strategy to DR or closely related graded medical imaging tasks. Note: a strategy validated on natural image classification (e.g., ImageNet) is weaker evidence for DR grading — ordinal labels and class imbalance change the loss landscape.
3. **Interaction effects**: Identify known interactions with other training choices:
   - Does this loss interact with the choice of backbone pre-training?
   - Does it assume a particular label quality? (relevant given APTOS noise)
   - Does it require tuning additional hyperparameters? What is the sensitivity?
4. **Dataset-specific considerations**: Assess against Messidor-2 and APTOS 2019 characteristics specifically:
   - Messidor-2: adjudicated labels, relatively clean, moderate size (1,748)
   - APTOS 2019: Kaggle-sourced, known label noise, moderate size (3,662), class imbalance
5. **Verdict**: Same five-point scale as Type A. Add a `Recommended experiment` — one concrete ablation that would confirm or refute the strategy.

### Type C — Literature gap / research direction

The user proposes a research direction or claims a gap exists in the literature (e.g., "nobody has applied self-supervised pre-training specifically to fundus images for DR grading").

**Evaluation protocol:**

1. **Gap verification**: Do not accept a claimed gap without searching. Run at least two targeted searches. If papers exist that address the gap, surface them immediately. If no papers are found, state `[No papers found as of search date]` — not "this is a gap" as a definitive fact, since absence of evidence in a search is not proof of absence in the literature.
2. **Significance assessment**: If the gap is real, evaluate whether it matters:
   - Is there a plausible reason this direction would improve on current SOTA for DR grading?
   - Is the gap due to lack of interest, or due to known obstacles (data, compute, clinical constraints)?
3. **Neighbouring work**: Identify the closest existing work and quantify how far the proposed direction departs from it. A gap adjacent to strong prior work is more credible than one in an isolated corner.
4. **Feasibility**: Assess against realistic constraints — Messidor-2 and APTOS 2019 are moderate-scale datasets. Methods requiring very large pre-training corpora or extensive annotation may be infeasible without additional data.
5. **Output**: Produce a structured gap report (see Section 4).

---

## 2. Evidence Hierarchy

Rank evidence by strength when building an evaluation argument. Always cite the tier alongside the evidence.

| Tier | Description | Example |
|---|---|---|
| T1 — Direct | Result from a paper on DR severity grading using fundus images | Gulshan et al. (2016): AUC 0.99 on Messidor-2 for referable DR |
| T2 — Adjacent domain | Result from graded medical imaging (e.g., OCT, skin lesion, chest X-ray) | Ordinal loss improves QWK on ISIC skin grading (cite) |
| T3 — General DL | Result from natural image classification or detection | Focal loss reduces class imbalance effect on COCO (Lin et al., 2017) |
| T4 — Theoretical | Derivation or mathematical argument with no empirical DR validation | Ordinal CE has lower Lipschitz constant than standard CE [Inference] |
| T5 — Inference | First-principles reasoning, no citation | [Inference] Larger effective receptive field may capture haemorrhage context |

When an argument relies on T3 or below, explicitly flag the evidence gap: state what T1 or T2 evidence would be needed to strengthen it.

---

## 3. Handling Uncertainty

When evidence is mixed or absent, apply this protocol:

1. **State the evidential situation clearly** — which claims are supported, which are not, which are contested.
2. **Reason from first principles if warranted** — label every step as `[Inference]` and show the reasoning chain explicitly.
3. **Do not manufacture confidence** — if the honest answer is "we don't know", say so. A well-framed unknown is more useful than a false verdict.
4. **Propose a concrete test** — for every unresolved question, suggest one experiment or literature search that would reduce the uncertainty.

**Example of correct uncertainty handling:**

> The effect of label smoothing on QWK for DR grading has not been directly studied [No T1/T2 evidence found]. However, label smoothing reduces over-confidence on noisy labels (Müller et al., 2019 — T3), and APTOS 2019 is known to contain label noise. [Inference] It is plausible that label smoothing would improve calibration and QWK on APTOS, but the interaction with ordinal label structure is unknown. The recommended test is an ablation comparing CE, CE+label smoothing, and ordinal CE on APTOS 2019 with QWK and ECE as metrics.

---

## 4. Output Formats

### 4a. Single idea evaluation

Use this structure:

---

**Idea**: [Restate the idea precisely, in one sentence. If the user's statement was ambiguous, ask for clarification before proceeding.]

**Problem addressed**: [What limitation of current DR grading does this target?]

**Prior art**:
- [Paper 1 — what they did, what result they got, evidence tier]
- [Paper 2 — ...]
- If none found: `No directly relevant prior art found [search date].`

**Mechanism analysis**:
[Why this might work — ground each claim in either evidence or labelled inference.]

**Potential failure modes**:
- [Failure mode 1 — grounded in a known phenomenon or cited paper]
- [Failure mode 2 — ...]

**Dataset-specific notes** (Messidor-2 / APTOS 2019):
[How the specific characteristics of these datasets affect the idea's viability.]

**Verdict**: [Supported / Plausible / Contested / Unsupported]
[One paragraph justification.]

**Recommended experiment**:
[One concrete ablation or pilot study. Specify: baseline, proposed variant, dataset, primary metric.]

---

### 4b. Literature gap report

---

**Claimed gap**: [Restate the gap as a falsifiable statement.]

**Search results**:
- Queries run: [list queries]
- Papers found addressing this gap (if any): [list with citations]
- Conclusion: [Gap confirmed / Partially addressed / Gap does not exist — papers exist]

**Significance**:
[Why does this gap matter for DR grading specifically?]

**Obstacles explaining the gap** (if real):
[Data constraints, annotation cost, clinical deployment barriers, etc. — cite where possible.]

**Neighbouring work** (closest existing papers):
[What exists; how far the proposed direction departs from it.]

**Proposed research question**:
[Frame the gap as a single, falsifiable research question.]

**Feasibility on available benchmarks**:
[Can this be tested on Messidor-2 and/or APTOS 2019 alone, or does it require additional data?]

---

## 5. What This Skill Will Not Do

- **Not produce results without sources.** If a specific claim requires a number (AUC, QWK, sensitivity) it must come from a cited paper — not from general knowledge or estimation.
- **Not confirm ideas to be polite.** If an idea has significant problems, state them clearly. The goal is a publishable, defensible research direction, not validation.
- **Not evaluate outside the scope of DR severity grading** without flagging the scope change. If you ask about lesion detection or vessel segmentation, the evaluation will note that it is operating outside the primary scope and evidence from the DR grading literature may not transfer.

---

## 6. DR Grading Context (Shared with research.md)

Key benchmarks and their properties — always factor these into evaluations:

| Dataset | Size | Label quality | Class imbalance | Primary metric |
|---|---|---|---|---|
| Messidor-2 | 1,748 | Adjudicated — high quality | Moderate | AUC (referable) |
| APTOS 2019 | 3,662 | Kaggle crowd — noisy | High (no DR dominant) | QWK |

**ICDR grading scale**: 0 (No DR) → 1 (Mild) → 2 (Moderate) → 3 (Severe) → 4 (Proliferative). Ordinal structure matters — methods that treat grades as independent classes ignore this structure.

**Referable threshold**: Grades ≥ 2 are standard for referable DR. Some studies use ≥ 1. When comparing sensitivity/specificity across papers, always check which threshold was used.

**Class imbalance**: In most DR datasets, grade 0 (No DR) is the majority class, often 50–70% of images. Grade 3 and 4 are rare. This affects loss function choice, augmentation strategy, and how accuracy should be interpreted.