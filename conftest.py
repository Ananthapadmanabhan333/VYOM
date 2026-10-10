import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"

for d in [str(root_dir), str(src_dir)]:
    if d not in sys.path:
        sys.path.insert(0, d)
