"""Quick syntax + import validation for modified files."""
import ast
import sys
from pathlib import Path

files = [
    "src/config.py",
    "src/dataset.py",
    "src/train.py",
]

errors = 0
for f in files:
    try:
        source = Path(f).read_text(encoding="utf-8")
        ast.parse(source, filename=f)
        print(f"  [OK] {f} — syntax valid")
    except SyntaxError as e:
        print(f"  [FAIL] {f} — {e}")
        errors += 1

if errors:
    print(f"\n{errors} file(s) have syntax errors!")
    sys.exit(1)
else:
    print("\nAll files pass syntax check.")
    sys.exit(0)
