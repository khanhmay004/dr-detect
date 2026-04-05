# Plan 08 — Augmentation-Based Class Balancing

> **Status**: Run D in progress (2026-04-05)  
> **Spec doc**: `docs/08-aug-improv.md`  
> **Original baseline**: `baseline_resnet50_20260403_120248` (Messidor-2 QWK: 0.6233)

---

## Summary

Plan 08 addresses minority class (Grade 1, 2) recall collapse on Messidor-2 through:
1. **Quantity balancing** — Oversample minority classes to 800/class via `AugmentedBalancedDataset`
2. **Class-conditional augmentation** — Conservative transforms based on published DR paper
3. **Alpha adjustment** — Always use inverse-frequency focal loss alpha

---

## Experimental Results (Runs B & C — FAILED)

### Run B — Quantity + Uniform Alpha

**Checkpoint**: `baseline_resnet50_20260405_052029_fold0`  
**Config**: `aug_focal_alpha_uniform: true`, standard transforms for all grades

| Metric | Original | Run B | Δ |
|--------|----------|-------|---|
| **Messidor-2 QWK** | **0.6233** | 0.5043 | −19% |
| Grade 0 Recall | 0.942 | 0.988 | ↑ |
| Grade 1 Recall | 0.093 | **0.000** | ❌ |
| Grade 2 Recall | 0.161 | 0.112 | ↓ |

### Run C — Aggressive Transforms + Inverse Alpha

**Checkpoint**: `baseline_resnet50_20260405_070645_fold0`  
**Config**: `aug_focal_alpha_uniform: false`, aggressive heavy/medium transforms

| Metric | Original | Run C | Δ |
|--------|----------|-------|---|
| **Messidor-2 QWK** | **0.6233** | 0.4957 | −20% |
| Grade 1 Recall | 0.093 | **0.007** | ❌ |
| Grade 2 Recall | 0.161 | **0.014** | ❌ |

**Root cause**: Aggressive augmentation (ElasticTransform, CoarseDropout, high hue shift) destroyed subtle Grade 1/2 lesion features.

---

## Run D — Conservative Paper-Based Augmentation (IN PROGRESS)

### Code Changes Applied (2026-04-05)

Replaced aggressive transforms with conservative pipeline based on published DR paper:

**Old `_build_heavy_transform()` (REMOVED):**
- ElasticTransform, GridDistortion
- CoarseDropout (1-4 holes)
- HueSaturationValue (hue=15°)
- Rotation ±90°
- MotionBlur, ImageCompression

**New `_build_heavy_transform()` (IMPLEMENTED):**
```python
def _build_heavy_transform(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Conservative augmentation for Grades 1/2 — based on published DR paper."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5
        ),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            p=0.5,
        ),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            shear=(-10, 10),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 2.0), p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        A.Perspective(scale=(0.05, 0.2), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
```

**New `_build_medium_transform()` (IMPLEMENTED):**
```python
def _build_medium_transform(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Medium augmentation for Grades 3/4 — less aggressive than heavy."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03, p=0.4
        ),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.85, 1.15),
            p=0.4,
        ),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            shear=(-5, 5),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.4,
        ),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 1.5), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
```

### Key Differences: Paper vs Old Aggressive

| Transform | Paper-Based | Old Aggressive | Impact |
|-----------|-------------|----------------|--------|
| Hue shift | **0.05** | 0.15 | Preserves red/green DR contrast |
| Rotation | **±25°** | ±90° | Maintains lesion orientation |
| Elastic | **None** | alpha=1.0 | No microaneurysm distortion |
| Dropout | **None** | 1-4 holes | No random lesion masking |
| Perspective | **0.2 @ 30%** | None | Mild camera angle variation |
| Sharpen | **Yes** | No | Enhances lesion edges |

### Test Commands

```bash
# On GPU instance — ensure inverse-freq alpha
sed -i 's/aug_focal_alpha_uniform: true/aug_focal_alpha_uniform: false/' configs/gpu_aug_balanced.yaml

# Train Run D
cd src
python train.py \
  --config configs/gpu_aug_balanced.yaml \
  --fold 0

# Evaluate on Messidor-2
python evaluate.py \
  --checkpoint ../outputs/checkpoints/baseline_resnet50_*_best.pth \
  --dataset messidor2 \
  --mc_passes 20 \
  --model baseline
```

### Success Criteria for Run D

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Messidor-2 QWK | > 0.50 | Better than failed Run C |
| Messidor-2 Grade 0 Recall | ≥ 85% | Guard rail |
| Messidor-2 Grade 1 Recall | > 9.3% | Beat original baseline |
| Messidor-2 Grade 2 Recall | > 16.1% | Beat original baseline |

---

## Bug Analysis (from Runs B & C)

### Issue 1: Aggressive Augmentation Destroys Lesion Features

Grade 1/2 features (microaneurysms, small hemorrhages) are position-sensitive and color-sensitive. ElasticTransform, CoarseDropout, and high hue shift destroyed these signals.

**Fix applied**: Conservative paper-based transforms.

### Issue 2: Uniform Alpha Catastrophic

Uniform alpha `[1,1,1,1,1]` assumes balanced test set. Messidor-2 is 1017:270 (Grade 0:1). Model maximizes accuracy by predicting Grade 0.

**Fix applied**: Always use inverse-frequency alpha.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/config.py` | +4 constants |
| `src/dataset.py` | +AugmentedBalancedDataset, +3 transform factories (now conservative) |
| `src/configs/experiment_config.py` | +3 dataclass fields |
| `src/train.py` | +3 CLI args, updated alpha logic |
| `src/configs/gpu_aug_balanced.yaml` | Config with `aug_focal_alpha_uniform: false` |

---

## Next Steps

1. **Run D**: Train with conservative augmentation + inverse-freq alpha
2. **Evaluate**: Messidor-2 + APTOS test
3. **If successful**: Multi-fold training (folds 1-4)
4. **If failed**: Abandon augmentation approach, investigate domain adaptation
