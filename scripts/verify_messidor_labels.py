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