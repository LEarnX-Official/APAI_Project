"""Print the dataset composition — the numbers for the paper's data section."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import get_config, CLASSES  # noqa: E402


def main():
    cfg = get_config()
    root = Path(cfg.data.root)
    if not root.is_dir():
        raise SystemExit(f"dataset not found at {root}")

    rows, totals = [], {"train": 0, "test": 0}
    for cls in CLASSES:
        counts = {}
        for split in ("train", "test"):
            d = root / split / cls
            counts[split] = sum(
                1 for f in d.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            ) if d.is_dir() else 0
            totals[split] += counts[split]
        rows.append((cls, counts["train"], counts["test"]))

    width = max(len(c) for c in CLASSES) + 2
    print(f"{'class':<{width}}{'train':>8}{'test':>8}")
    print("-" * (width + 16))
    for cls, tr, te in rows:
        print(f"{cls:<{width}}{tr:>8}{te:>8}")
    print("-" * (width + 16))
    print(f"{'total':<{width}}{totals['train']:>8}{totals['test']:>8}")

    tr_counts = [r[1] for r in rows]
    if min(tr_counts) > 0:
        print(f"\nimbalance ratio (max/min, train): "
              f"{max(tr_counts) / min(tr_counts):.1f}:1")


if __name__ == "__main__":
    main()
