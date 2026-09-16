"""Recompute the grayscale mean/std over the training split.

The values in config.DataConfig come from this script. Rerun it if the dataset
is ever swapped or resampled, and paste the output back into config.py.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import get_config, CLASSES  # noqa: E402


def main():
    cfg = get_config()
    root = Path(cfg.data.root) / "train"
    if not root.is_dir():
        raise SystemExit(f"training split not found at {root}")

    # Streaming sum / sum-of-squares: avoids holding 28k images in memory.
    total, total_sq, n = 0.0, 0.0, 0
    for cls in CLASSES:
        d = root / cls
        if not d.is_dir():
            raise SystemExit(f"missing class directory {d}")
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            a = np.asarray(Image.open(f).convert("L"), dtype=np.float64) / 255.0
            total += a.sum()
            total_sq += (a ** 2).sum()
            n += a.size

    if n == 0:
        raise SystemExit("no images found")

    mean = total / n
    std = (total_sq / n - mean ** 2) ** 0.5

    print(f"pixels examined: {n:,}")
    print(f"mean = {mean:.4f}")
    print(f"std  = {std:.4f}")
    print("\nPaste into src/config.py DataConfig:")
    print(f"    mean: float = {mean:.4f}")
    print(f"    std: float = {std:.4f}")

    current = (cfg.data.mean, cfg.data.std)
    if abs(current[0] - mean) < 5e-4 and abs(current[1] - std) < 5e-4:
        print("\nconfig.py is already up to date.")
    else:
        print(f"\nconfig.py currently has mean={current[0]}, std={current[1]}"
              " — update it.")


if __name__ == "__main__":
    main()
