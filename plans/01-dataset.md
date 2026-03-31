# Messidor-2 Dataset Investigation: Label Problem Analysis & Fix

> **Severity**: 🔴 **CRITICAL** — The current pipeline is configured to use **wrong labels** for Messidor-2 evaluation.
> **Date**: 2026-03-31
> **Status**: Code fixes complete ✅ — awaiting data download & verification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Investigation Findings](#3-investigation-findings)
4. [Root Cause Analysis](#4-root-cause-analysis)
5. [Impact Assessment](#5-impact-assessment)
6. [Recommended Fix](#6-recommended-fix)
7. [Code Changes Required](#7-code-changes-required)
8. [Verification Plan](#8-verification-plan)
9. [Thesis Documentation Impact](#9-thesis-documentation-impact)
10. [References](#10-references)

---

## 1. Executive Summary

The Messidor-2 external evaluation pipeline has a **fundamental data integrity problem**: the `trainLabels.csv` file currently in the `messidor-2/` directory **does not contain Messidor-2 labels**. It contains labels from the **Kaggle EyePACS Diabetic Retinopathy Detection competition** (35,127 entries), which is an entirely different dataset.

The project has the correct Messidor-2 **images** (1,748 files in `IMAGES/`), but lacks the correct **DR grade labels** needed for evaluation. The official Messidor-2 dataset was released without DR severity grades — these must be obtained from a third-party source.

> ⚠️ **CRITICAL**: Running `evaluate.py` with the current `trainLabels.csv` will attempt to match EyePACS image IDs (like `10_left`, `10_right`) against Messidor-2 image filenames (like `20051109_57451_0400_PP.png`, `IM000013.JPG`). **No matches will align correctly**, and any results produced would be scientifically invalid.

---

## 2. Problem Statement

### What the user reported
> "The official Messidor-2 does not include DR grades — labels come from third-party sources."

### What we found
The problem is actually **two-fold** and more severe than initially described:

1. **The `trainLabels.csv` is from the wrong dataset entirely** — it belongs to EyePACS/Kaggle DR competition, not Messidor-2.
2. **No valid Messidor-2 DR grade labels exist anywhere in the project** — the `messidor-2.csv` file that does ship with the official dataset contains only left/right image pair mappings, not DR grades.

---

## 3. Investigation Findings

### 3.1. Files in `messidor-2/` directory

| File | Size | Contents | Source |
|------|------|----------|--------|
| `trainLabels.csv` | 465 KB (35,128 lines) | `image,level` — EyePACS competition labels | ❌ **Wrong dataset** (Kaggle EyePACS) |
| `messidor-2.csv` | 38 KB (876 lines) | `left;right` — image pair mappings only | ✅ Official Messidor-2 (but no DR grades) |
| `sampleSubmission.csv` | 710 KB (53,578 lines) | `image,level` — all zeros (template) | ❌ **Wrong dataset** (Kaggle EyePACS submission template) |
| `IMAGES/` | ~1,748 files | `.png` and `.JPG` fundus images | ✅ Official Messidor-2 images |

### 3.2. Detailed CSV Analysis

#### `trainLabels.csv` — ❌ EyePACS Labels (NOT Messidor-2)

```
image,level
10_left,0
10_right,0
13_left,0
15_left,1
15_right,2
16_left,4
...
```

- **35,127 data rows** + 1 header = 35,128 lines
- Image ID format: `{patient_number}_{left|right}` (e.g., `10_left`, `10_right`)
- Level: 0–4 (ICDR DR grades)
- **This is the standard EyePACS Kaggle competition training labels file** (~35K images)
- None of these image IDs match any filename in `IMAGES/`

#### `messidor-2.csv` — ✅ Official Pair Mapping (No DR Grades)

```
left;right 
20051109_57451_0400_PP.png;20051109_57843_0400_PP.png
20051109_59864_0400_PP.png;20051109_59751_0400_PP.png
...
IM000013.JPG;IM000012.JPG
IM000057.JPG;IM000056.JPG
...
```

- **875 data rows** + 1 header = 876 lines (= 875 examinations, each with left + right = 1,750 images; minus 2 for potential duplicates → 1,748 unique images)
- Semicolon-delimited (`;`), not comma-delimited
- Contains **only image filenames** paired as left/right — **zero DR grade information**
- Two naming conventions:
  - `YYYYMMDD_NNNNN_NNNN_PP.png` — French hospital acquisitions (high-res, ~1–3 MB each)
  - `IMNNNNNN.JPG` — from the original Messidor-1 subset (~100–160 KB each)

#### `sampleSubmission.csv` — ❌ EyePACS Submission Template

```
image,level
1_left,0
1_right,0
2_left,0
...
```

- **53,577 data rows** — all levels are `0` (placeholder template for Kaggle submission)
- This is the Kaggle DR competition's test-set submission template
- Completely irrelevant to Messidor-2

### 3.3. IMAGES Directory Analysis

- **1,748 image files** — matches the official Messidor-2 count (874 examinations × 2 eyes)
- Two image types present:
  - **~1,060 PNG files** — `YYYYMMDD_NNNNN_NNNN_PP.png` naming (from Brest University Hospital + extension)
  - **~688 JPG files** — `IMNNNNNN.JPG` naming (from original Messidor-1)
- All files are accounted for in `messidor-2.csv` pair mappings

### 3.4. Current Code Configuration

From `src/config.py` (line 27):
```python
MESSIDOR_CSV = MESSIDOR_DIR / "trainLabels.csv"  # Contains 'image' and 'level' columns
```

From `src/dataset.py` (lines 182–203):
```python
# MessidorDataset.__getitem__:
image_id = str(row["image"])          # ← reads 'image' column from trainLabels.csv
...
if self.labels_available and "level" in self.df.columns:
    label = int(row["level"])          # ← reads 'level' column from trainLabels.csv
```

From `src/evaluate.py` (lines 290–296):
```python
dataset = MessidorDataset(
    csv_path=str(MESSIDOR_CSV),        # ← points to trainLabels.csv
    image_dir=str(MESSIDOR_IMAGES),
    ...
)
```

From `docs/research-project.md` (lines 234–243):
```
| Property      | Value                                                           |
| ------------- | --------------------------------------------------------------- |
| Total images  | 690 (filtered `adjudicated_dr_grade` subset)                    |
| Label quality | Adjudicated by multiple ophthalmologists — cleaner ground truth |
```

> ⚠️ **WARNING**: The research document references `adjudicated_dr_grade` and claims 690 images, but no file in the project contains an `adjudicated_dr_grade` column. The 690 number likely comes from the subset of fully-gradable images with adjudicated labels from the Krause et al. (2018) dataset — which has **never been downloaded**.

---

## 4. Root Cause Analysis

The dataset was likely obtained from a Kaggle page that bundles both EyePACS and Messidor-2 under a single "Diabetic Retinopathy" umbrella. Common Kaggle pages include:

1. **The original Kaggle DR Detection competition** — provides `trainLabels.csv` (EyePACS) and `sampleSubmission.csv`
2. **The Messidor-2 image collection** — provides `messidor-2.csv` (pair mappings only) and `IMAGES/`

When downloaded together, both datasets end up in the same directory. The `trainLabels.csv` from EyePACS got incorrectly associated with Messidor-2 images.

### Why this wasn't caught earlier

1. The `trainLabels.csv` has the expected column names (`image`, `level`) that the code looks for
2. The code silently loads the CSV without validating that image IDs actually exist in `IMAGES/`
3. CPU smoke tests and APTOS training don't touch the Messidor-2 pipeline
4. GPU evaluation on Messidor-2 has been deferred until Phase 1.5 (not yet reached)

---

## 5. Impact Assessment

### 5.1. What is broken

| Component | Status | Impact |
|-----------|--------|--------|
| `MessidorDataset` loading | ❌ **Will crash** | `_find_image_path()` will throw `FileNotFoundError` for every image ID like `10_left` |
| `evaluate.py` end-to-end | ❌ **Will crash** | Cannot complete MC Dropout inference |
| Messidor-2 metrics (QWK, AUC, Acc) | ❌ **Impossible** | No valid labels available |
| Cross-domain generalization claim | ❌ **Unsupported** | Cannot validate APTOS → Messidor-2 without labels |
| Thesis Section 4 (Experimental Results) | ⚠️ **Blocked** | External validation is a key thesis deliverable |

### 5.2. What is NOT broken

| Component | Status | Notes |
|-----------|--------|-------|
| APTOS training pipeline | ✅ OK | Completely separate data path |
| Model architecture (CBAM, baseline) | ✅ OK | No data dependency |
| Messidor-2 images | ✅ OK | All 1,748 images present and valid |
| `messidor-2.csv` pair mapping | ✅ OK | Correct official file |

---

## 6. Recommended Fix

### 6.1. Obtain correct labels: **Kaggle google-brain/messidor2-dr-grades**

The adjudicated DR grades for Messidor-2 are available from:

📦 **[kaggle.com/datasets/google-brain/messidor2-dr-grades](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades)**

This dataset provides a CSV (typically named `messidor_data.csv` or similar) with the following columns:

| Column | Description |
|--------|-------------|
| `image_id` | Messidor-2 image filename (matches files in `IMAGES/`) |
| `adjudicated_dr_grade` | 5-point ICDR DR grade (0–4) adjudicated by 3 retina specialists |
| `adjudicated_dme` | DME grade (not needed for this project) |
| `adjudicated_gradable` | Whether the image was deemed gradable (important for filtering) |

**Citation required** (add to thesis references):
> Krause, J. et al. "Grader Variability and the Importance of Reference Standards for Evaluating Machine Learning Models for Diabetic Retinopathy." *Ophthalmology*, 2018.

### 6.2. Why this source is appropriate

1. **Expert adjudication**: Labels come from a panel of **3 US board-certified retina specialists** (not a single Kaggle grader like APTOS)
2. **Widely used benchmark**: This is the standard Messidor-2 label source used by Gulshan et al. (2016), Krause et al. (2018), Voets et al. (2019), and many recent papers
3. **Cleaner than APTOS**: The research document already acknowledges this: "Adjudicated by multiple ophthalmologists — cleaner ground truth" (line 242)
4. **Gradability filtering**: The `adjudicated_gradable` column allows filtering out ungradable images (e.g., closed eyes, severe cataracts, out of focus), which is standard practice

### 6.3. Expected data characteristics after fix

| Property | Value |
|----------|-------|
| Total images in CSV | ~1,748 (all Messidor-2 images) |
| Gradable subset | ~1,700–1,748 (after filtering `adjudicated_gradable == 1`) |
| DR grade range | 0–4 (ICDR scale, same as APTOS) |
| Previous reference in docs | "690 filtered" — this may have been an older subset; verify after download |

> **IMPORTANT**: The "690 images" mentioned in `research-project.md` might be based on outdated information or a different filtering criterion. After downloading the correct labels, count the actual number of gradable images with valid DR grades and update the document accordingly.

### 6.4. What to do with existing wrong files

| File | Action |
|------|--------|
| `trainLabels.csv` | **DELETE** or rename to `eyepacs_trainLabels.csv.bak` — it belongs to EyePACS, not Messidor-2 |
| `sampleSubmission.csv` | **DELETE** — EyePACS competition template, irrelevant |
| `messidor-2.csv` | **KEEP** — legitimate official pair mapping file |
| `IMAGES/` | **KEEP** — correct Messidor-2 images |
| New `messidor_data.csv` | **ADD** — download from Kaggle google-brain dataset |

---

## 7. Code Changes Required

### 7.1. Download the correct labels

```bash
# Option 1: Kaggle CLI (if you have kaggle credentials configured)
kaggle datasets download -d google-brain/messidor2-dr-grades -p messidor-2/

# Option 2: Manual download from browser
# Go to: https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades
# Download and extract to messidor-2/
```

After download, verify the CSV structure. Expected format:
```csv
image_id,adjudicated_dr_grade,adjudicated_dme,adjudicated_gradable
20051020_43808_0100_PP.png,0,0,1
20051020_43832_0100_PP.png,2,1,1
...
```

### 7.2. Update `src/config.py`

```diff
- MESSIDOR_CSV = MESSIDOR_DIR / "trainLabels.csv"  # Contains 'image' and 'level' columns
+ MESSIDOR_CSV = MESSIDOR_DIR / "messidor_data.csv"  # Adjudicated DR grades (Krause et al. 2018)
```

> **NOTE**: The actual filename may vary depending on the Kaggle download. Common names: `messidor_data.csv`, `messidor_2_grades.csv`, or similar. Verify after download and adjust accordingly.

### 7.3. Update `src/dataset.py` — `MessidorDataset`

The `MessidorDataset` class needs to be updated to read the correct column names:

```python
class MessidorDataset(Dataset):
    """PyTorch Dataset for Messidor-2 fundus images (inference only).

    The Messidor-2 adjudicated grades CSV (Krause et al. 2018) uses
    ``image_id`` and ``adjudicated_dr_grade`` columns. Images may be
    ``.png`` or ``.JPG`` — we try all common extensions.

    If ``labels_available=False``, a dummy label of ``-1`` is returned
    (useful when running pure inference without ground truth).
    """

    # Column names for the adjudicated grades CSV
    COL_IMAGE_ID = "image_id"
    COL_DR_GRADE = "adjudicated_dr_grade"
    COL_GRADABLE = "adjudicated_gradable"

    def __init__(
        self,
        csv_path: str = str(MESSIDOR_CSV),
        image_dir: str = str(MESSIDOR_IMAGES),
        transform: A.Compose | None = None,
        preprocess: bool = True,
        target_size: int = IMAGE_SIZE,
        labels_available: bool = True,
        use_cache: bool = False,
        cache_dir: Path | None = None,
        filter_gradable: bool = True,  # NEW: filter ungradable images
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.preprocess = preprocess
        self.target_size = target_size
        self.labels_available = labels_available
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        # --- Detect CSV format and normalize column names ---
        self._normalize_columns()

        # --- Filter to gradable images only ---
        if filter_gradable and self.COL_GRADABLE in self.df.columns:
            n_before = len(self.df)
            self.df = self.df[self.df[self.COL_GRADABLE] == 1].reset_index(drop=True)
            n_after = len(self.df)
            if n_before != n_after:
                print(f"  Messidor-2: Filtered {n_before - n_after} ungradable images "
                      f"({n_before} -> {n_after})")

        # --- Validate: check that at least some image files exist ---
        self._validate_image_availability()

    def _normalize_columns(self):
        """Detect CSV format and normalize to standard column names.

        Supports multiple CSV formats:
        - Adjudicated format: image_id, adjudicated_dr_grade, adjudicated_gradable
        - Legacy format: image, level (from EyePACS — should be flagged)
        """
        cols = set(self.df.columns)

        if "image_id" in cols and "adjudicated_dr_grade" in cols:
            # Standard adjudicated format — no changes needed
            pass
        elif "image" in cols and "level" in cols:
            # Legacy EyePACS format — rename for compatibility but warn
            print("  WARNING: CSV uses legacy 'image'/'level' column names.")
            print("    This may be the EyePACS trainLabels.csv — verify data source!")
            self.df = self.df.rename(columns={
                "image": "image_id",
                "level": "adjudicated_dr_grade"
            })
        else:
            available = list(self.df.columns)
            raise ValueError(
                f"Messidor-2 CSV has unexpected columns: {available}. "
                f"Expected 'image_id' + 'adjudicated_dr_grade' "
                f"or 'image' + 'level'."
            )

    def _validate_image_availability(self, sample_n: int = 5):
        """Check that at least some image IDs actually exist in IMAGES/."""
        sample = self.df["image_id"].head(sample_n).tolist()
        found = 0
        for img_id in sample:
            try:
                self._find_image_path(str(img_id))
                found += 1
            except FileNotFoundError:
                pass
        if found == 0:
            raise RuntimeError(
                f"No Messidor-2 images found for the first {sample_n} IDs in CSV. "
                f"Sample IDs: {sample}. "
                f"Image directory: {self.image_dir}. "
                f"This likely means the CSV does not match the images — "
                f"check that you are using the correct Messidor-2 label file."
            )

    def __getitem__(self, idx: int):
        """Returns ``(image, label, image_id)``."""
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        # ... (rest of image loading logic unchanged) ...

        # Read DR grade from adjudicated column
        if self.labels_available and "adjudicated_dr_grade" in self.df.columns:
            label = int(row["adjudicated_dr_grade"])
        else:
            label = -1

        # ... (rest of transform logic unchanged) ...

        return image, label, image_id
```

**Key changes**:
1. Column names: `image` → `image_id`, `level` → `adjudicated_dr_grade`
2. Added `filter_gradable` parameter to exclude ungradable images
3. Added `_normalize_columns()` for backward compatibility + format detection
4. Added `_validate_image_availability()` as a safety check to catch mismatched CSVs early

### 7.4. Update `src/evaluate.py` (minor)

The `evaluate.py` should print the actual number of gradable images:

```python
# In main(), after dataset creation:
print(f"  Images: {len(dataset)}")
if hasattr(dataset, 'df') and 'adjudicated_gradable' in dataset.df.columns:
    print(f"  (filtered to gradable images only)")
```

### 7.5. Update `docs/research-project.md`

Update Section 7.2 with the correct information after downloading labels:

```markdown
### 7.2. Messidor-2 (External Test Set)

| Property      | Value                                                           |
| ------------- | --------------------------------------------------------------- |
| Source        | Decenciere et al. (images) + Krause et al. 2018 (labels)       |
| Total images  | 1,748 (874 examinations x 2 eyes)                              |
| Gradable      | ~X,XXX (after filtering adjudicated_gradable == 1)             |
| Format        | PNG / JPG                                                       |
| Role          | External domain generalization test                             |
| Label quality | Adjudicated by 3 US board-certified retina specialists          |
| Label source  | Kaggle google-brain/messidor2-dr-grades (Krause et al. 2018)   |
```

---

## 8. Verification Plan

### 8.1. After downloading correct labels

Run this verification script to confirm everything aligns:

```python
"""Verify Messidor-2 labels match images."""
import pandas as pd
from pathlib import Path

# Load new labels
csv_path = Path("messidor-2/messidor_data.csv")  # adjust filename if different
df = pd.read_csv(csv_path)

print(f"CSV shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# Check column names
assert "image_id" in df.columns, f"Missing 'image_id' column. Got: {list(df.columns)}"
assert "adjudicated_dr_grade" in df.columns, f"Missing 'adjudicated_dr_grade'. Got: {list(df.columns)}"

# Check grade distribution
print(f"\nDR Grade Distribution:")
print(df["adjudicated_dr_grade"].value_counts().sort_index())

# Check image existence
image_dir = Path("messidor-2/IMAGES")
found, missing = 0, 0
missing_examples = []
for img_id in df["image_id"]:
    img_id_str = str(img_id)
    # Try exact match and common extensions
    exists = False
    for ext in ["", ".png", ".tif", ".jpg", ".JPG", ".jpeg"]:
        if (image_dir / f"{img_id_str}{ext}").exists():
            exists = True
            break
    if (image_dir / img_id_str).exists():
        exists = True
    if exists:
        found += 1
    else:
        missing += 1
        if len(missing_examples) < 5:
            missing_examples.append(img_id_str)

print(f"\nImage Verification:")
print(f"  Found: {found}")
print(f"  Missing: {missing}")
if missing_examples:
    print(f"  Missing examples: {missing_examples}")

# Filter gradable
if "adjudicated_gradable" in df.columns:
    gradable = df[df["adjudicated_gradable"] == 1]
    print(f"\nGradable images: {len(gradable)} / {len(df)}")
    print(f"\nGradable DR Grade Distribution:")
    print(gradable["adjudicated_dr_grade"].value_counts().sort_index())

print("\n✓ Verification complete!")
```

### 8.2. Checklist

- [ ] Download `messidor_data.csv` (or equivalent) from Kaggle google-brain/messidor2-dr-grades
- [ ] Place in `messidor-2/` directory
- [ ] Run verification script above — confirm:
  - [ ] All `image_id` values match files in `IMAGES/`
  - [ ] `adjudicated_dr_grade` column exists with values 0–4
  - [ ] Gradable count is reasonable (expect ~1,700+)
- [x] Update `config.py`: `MESSIDOR_CSV = MESSIDOR_DIR / "messidor_data.csv"` *(done — line 27 of config.py)*
- [x] Update `dataset.py`: column names and gradable filtering *(done — `MessidorDataset` rewritten with `_normalize_columns()`, `_validate_image_availability()`, `filter_gradable` param, and `COL_IMAGE_ID`/`COL_DR_GRADE`/`COL_GRADABLE` class constants)*
- [x] Update `evaluate.py`: print gradable image count *(done — lines 304-306)*
- [ ] Delete or rename `trainLabels.csv` and `sampleSubmission.csv`
- [ ] Run `evaluate.py` end-to-end with a checkpoint (smoke test)
- [x] Update `docs/research-project.md` with correct Messidor-2 metadata *(done — Section 7.2 updated with Krause et al. source, gradable count, label provenance note; Issue 8 marked RESOLVED)*
- [x] Add Krause et al. 2018 citation to thesis references *(done — Reference #24)*
- [x] Fix `GaussNoise` API deprecation (Issue 6) *(done — uses `std_range` param)*

---

## 9. Thesis Documentation Impact

### 9.1. Sections that need updating

| Section | Change needed |
|---------|--------------|
| §7.2 Messidor-2 | Update image count from 690 → actual gradable count |
| §7.2 Messidor-2 | Add Krause et al. as label source |
| §7.2 Messidor-2 | Update "Label quality" to cite 3-specialist adjudication |
| §10.3 Evaluation Workflow | Mention gradable filtering step |
| §17 Key Limitations | Add note about third-party labels for Messidor-2 |
| §19 References | Add Krause et al. 2018 citation |

### 9.2. Suggested limitation statement

> **Messidor-2 label provenance**: The Messidor-2 dataset was released without DR severity grades. This project uses adjudicated labels from Krause et al. (2018), where each image was independently graded by three US board-certified retina specialists and adjudicated to consensus. While this is the standard approach in the literature (used by Gulshan et al. 2016, Voets et al. 2019), it means the "ground truth" for external validation is itself derived from expert opinion rather than a definitive clinical diagnosis. This is a common limitation across all Messidor-2 evaluation studies.

### 9.3. Citation to add

```
Krause, J. et al. "Grader Variability and the Importance of Reference Standards
for Evaluating Machine Learning Models for Diabetic Retinopathy."
Ophthalmology, vol. 125, no. 8, 2018, pp. 1264-1272.
doi:10.1016/j.ophtha.2018.01.034
```

---

## 10. References

1. **Official Messidor-2 dataset**: Decenciere, E. et al. "Feedback on a publicly distributed image database: the Messidor database." *Image Analysis & Stereology*, 2014. [ADCIS Distribution](https://www.adcis.net/en/third-party/messidor/)

2. **Adjudicated DR grades**: Krause, J. et al. "Grader Variability and the Importance of Reference Standards for Evaluating Machine Learning Models for Diabetic Retinopathy." *Ophthalmology*, 2018.

3. **Kaggle labels dataset**: [kaggle.com/datasets/google-brain/messidor2-dr-grades](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades)

4. **EyePACS competition** (source of the wrong `trainLabels.csv`): [kaggle.com/c/diabetic-retinopathy-detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)

5. **Reproducibility concerns with Messidor-2**: Voets, M. et al. "Reproduction Study Using Public Data of: Development and Validation of a Deep Learning Algorithm for Detection of DR in Retinal Fundus Photographs." *arXiv*, 2019.

---

> **Next step**: Download the correct labels from Kaggle, run the verification script, and apply the code changes above. This is a **blocking prerequisite** for Phase 1.5 (Messidor-2 evaluation).
