# Code Style Rules

> **Scope**: All Python code output. Load this file on every session alongside `AGENTS.md`.
> **Python version**: 3.12. Use modern syntax where it improves clarity (e.g., `match`, `X | Y` unions, `tomllib`).

---

## 1. Structure and Paradigm

**Keep files flat.** Do not create `controllers/`, `services/`, `utils/` subdirectories unless the project explicitly requires it and the user has approved the structure. One directory level of organisation is almost always enough.

**Default to functions, not classes.** Classes are permitted only when:
- Subclassing a framework type (e.g., `torch.nn.Module`, `torch.utils.data.Dataset`)
- The object genuinely manages lifecycle state that cannot be cleanly passed as arguments
- The user explicitly requests a class

When in doubt, write a function. A module full of pure functions is easier to test, debug, and reason about than a tangle of objects.

**Avoid global mutable state.** Pass configuration and state explicitly through function arguments.

---

## 2. Naming

Use **full, descriptive names** that encode the business rule or intent:

```python
# Correct
min_students_per_room: int
learning_rate_warmup_steps: int
patient_survival_probability: float

# Forbidden
min_s: int
lr_ws: int
p: float
```

**Permitted short names** (narrow, well-understood contexts only):

| Name | Permitted context |
|---|---|
| `i`, `j`, `k` | Loop indices |
| `x`, `y`, `z` | Mathematical coordinates |
| `n`, `m` | Matrix dimensions in a formula |
| `df` | A single, unambiguous DataFrame in a short function |
| `t` | Timestep in a temporal loop |

Everything else gets a full name. If you are unsure, write the full name.

---

## 3. Typing

**All function signatures require type hints.** No exceptions.

```python
# Correct
def compute_class_weights(
    label_counts: pd.Series,
    smoothing_factor: float = 1.0,
) -> np.ndarray:
    ...

# Forbidden
def compute_class_weights(label_counts, smoothing_factor=1.0):
    ...
```

Use Python 3.12 union syntax (`X | Y`) over `Optional[X]` or `Union[X, Y]`:

```python
def load_checkpoint(path: str | None = None) -> dict | None:
    ...
```

Use `typing` for complex generics (`TypeVar`, `Protocol`, `Callable`, `TypedDict`) but avoid `typing` imports where built-in generics work: `list[int]`, `dict[str, float]`, `tuple[int, ...]`.

Do **not** use Pydantic for general-purpose runtime validation. Reserve it for API boundary validation only if the user explicitly introduces it.

---

## 4. Vectorisation vs. Loops

**For numerical / tensor / array operations: vectorisation is mandatory.** Loops over elements are forbidden.

```python
# Correct — vectorised
scaled_logits = raw_logits / temperature

# Forbidden — element-wise loop
for i in range(len(raw_logits)):
    scaled_logits[i] = raw_logits[i] / temperature
```

Permitted loop patterns:
- Iterating over files, dataset splits, experiment configs, or other non-numerical sequences
- Epoch/batch training loops
- Any case where a loop materially improves readability and the data is not a numeric array

When a loop over numerical data is genuinely clearer, add a comment explaining why vectorisation was avoided.

---

## 5. PyTorch Conventions

### Shape comments

Shape annotations are **mandatory** after any operation that changes tensor dimensions.

```python
embeddings = model.encode(token_ids)  # [B, S, D]
pooled = embeddings.mean(dim=1)       # [B, D]
logits = projection(pooled)           # [B, num_classes]
```

Use consistent single-letter conventions: `B` = batch, `S` = sequence, `D` = hidden dim, `C` = channels, `H` = height, `W` = width, `N` = arbitrary count, `T` = timesteps.

### Device discipline

Never hardcode `"cuda"` or `"cpu"`. Always resolve device from the model or a config:

```python
device = next(model.parameters()).device
tensor = tensor.to(device)
```

### Module structure

Keep `nn.Module` subclasses minimal: `__init__` defines layers, `forward` defines the computation. Do not add training logic, data loading, or logging inside a module.

### Sparse tensors

Handle sparse formats explicitly. Never call `.to_dense()` accidentally inside a loop or a hot path. If densification is required, do it once and document why.

---

## 6. Pandas Conventions

Use Pandas as the default for all tabular data. Do not use plain `dict` or `list` structures when the data is inherently tabular.

Prefer method chaining for multi-step transformations:

```python
# Correct
cleaned_df = (
    raw_df
    .dropna(subset=["patient_id", "diagnosis_date"])
    .rename(columns={"diag_dt": "diagnosis_date"})
    .assign(age_at_diagnosis=lambda df: df["diagnosis_date"].dt.year - df["birth_year"])
    .query("age_at_diagnosis >= 0")
)

# Avoid — mutating the same variable repeatedly
df.dropna(subset=["patient_id"], inplace=True)
df.rename(columns={"diag_dt": "diagnosis_date"}, inplace=True)
```

Do **not** iterate over DataFrame rows with `iterrows()` for numerical operations. Use vectorised column operations or `numpy` directly.

---

## 7. Formatting and Linting

| Tool | Config |
|---|---|
| `black` | `line-length = 88` |
| `flake8` | `max-line-length = 88`, `extend-ignore = E203, W503` |

All code must pass both before being presented. Apply `black` formatting mentally — consistent indentation, trailing commas in multi-line structures, no unnecessary parentheses.

**Import ordering** (enforced by `isort`, compatible with `black`):

```python
# 1. Standard library
import logging
from pathlib import Path

# 2. Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 3. Local / project
from src.data import load_dataset
```

No star imports (`from module import *`). Ever.

---

## 8. Configuration and I/O

All configurable values (paths, hyperparameters, credentials, thresholds) live in `.env` files. Load them at the top of the entry point using `python-dotenv`. Never hardcode magic strings or numbers that represent configuration.

```python
# Correct
from dotenv import load_dotenv
import os

load_dotenv()
data_root = Path(os.environ["DATA_ROOT"])

# Forbidden
data_root = Path("/home/user/projects/data")
```

**Use `pathlib.Path` exclusively** for all filesystem operations. `os.path` is forbidden.

```python
# Correct
output_path = Path("results") / "experiment_01" / "metrics.csv"

# Forbidden
output_path = os.path.join("results", "experiment_01", "metrics.csv")
```

---

## 9. Concurrency

Use `threading` for I/O-bound tasks (API calls, file downloads, database queries). Do not use `asyncio`.

For CPU-bound parallel work (e.g., data preprocessing), use `concurrent.futures.ProcessPoolExecutor`.

---

## 10. Error Handling (LBYL)

**Look Before You Leap.** Validate state and inputs with explicit `if` checks before acting. Do not use `try/except` for control flow.

```python
# Correct — LBYL
def load_model_weights(checkpoint_path: Path, model: nn.Module) -> nn.Module | None:
    if not checkpoint_path.exists():
        logging.error("Checkpoint not found: %s", checkpoint_path)
        return None
    if not checkpoint_path.suffix == ".pt":
        logging.error("Expected .pt file, got: %s", checkpoint_path.suffix)
        return None
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model

# Forbidden — exception as control flow
def load_model_weights(checkpoint_path: Path, model: nn.Module) -> nn.Module | None:
    try:
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        return model
    except FileNotFoundError:
        return None
```

**Failure modes:**

| Error type | Action |
|---|---|
| Critical system error (corrupted state, unrecoverable) | Raise `Exception` immediately with a clear message |
| Logic / data error (bad input, missing file, unexpected value) | Log with `logging.error(...)` and return `None` |

Use `logging` throughout. Never use `print()` for diagnostic output in non-script code.

---

## 11. Documentation (Google Style)

Every function requires a docstring with: `Args`, `Returns`, and `Raises` sections, plus at least one usage example.

```python
def normalise_feature_matrix(
    feature_matrix: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Normalise a feature matrix to zero mean and unit variance.

    Applies standardisation along the specified axis. Uses population
    statistics (ddof=0), consistent with scikit-learn's StandardScaler
    default behaviour.

    Args:
        feature_matrix: Input array of shape [N, D] where N is the number
            of samples and D is the number of features.
        axis: Axis along which to compute mean and std. Defaults to 0
            (normalise across samples, per feature).

    Returns:
        Normalised array of the same shape as feature_matrix.

    Raises:
        ValueError: If any feature column has zero standard deviation
            (would produce NaN after division).

    Example:
        >>> features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> normalise_feature_matrix(features)
        array([[-1.22474487, -1.22474487],
               [ 0.        ,  0.        ],
               [ 1.22474487,  1.22474487]])
    """
```

**Mandatory inline comments** for:
- Non-obvious algorithmic choices (explain *why*, not *what*)
- Tensor shape transformations (see Section 5)
- Any place where a "naïve" implementation was deliberately avoided
- Changes to existing code: one-line comment explaining the reason for the change

When implementing from a paper, reference the equation number:

```python
# KL divergence term — Eq. (3) in Kingma & Welling (2013)
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

---

## 12. Testing

No test framework is mandated yet. When tests are introduced, default to `pytest`. Write tests as plain functions (not `unittest.TestCase` subclasses) and name them `test_<what_is_being_tested>`.

Until a framework is chosen, ensure all functions with non-trivial logic include a usage example in the docstring (see Section 11) that can double as a manual sanity check.

---

## Quick Reference

| Rule | Decision |
|---|---|
| Paradigm | Functional; classes only when required by framework |
| Structure | Flat files, no deep module hierarchy |
| Naming | Full descriptive names; abbreviations only in narrow contexts |
| Typing | Mandatory on all signatures; `X \| Y` union syntax |
| Vectorisation | Mandatory for all numerical operations |
| Formatting | `black` 88, `flake8` 88 |
| Paths | `pathlib` only; `os.path` forbidden |
| Config | `.env` + `python-dotenv`; no hardcoded values |
| Concurrency | `threading` for I/O; no `asyncio` |
| Error handling | LBYL; `logging` not `print`; return `None` on data errors |
| Docstrings | Google style; `Args`, `Returns`, `Raises`, `Example` required |
| Python version | 3.12 |