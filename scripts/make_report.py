"""Collect every finished run in results/ into paper-ready markdown tables."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import get_config  # noqa: E402
from evaluate import markdown_table  # noqa: E402


def pct(x):
    return f"{100 * x:.2f}"


def main():
    cfg = get_config()
    results_dir = Path(cfg.results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"no results directory at {results_dir}")

    baselines, maml_runs = [], []
    for mf in sorted(results_dir.glob("*/metrics.json")):
        blob = json.loads(mf.read_text())
        tag = mf.parent.name
        if "metrics" in blob:
            m = blob["metrics"]
            baselines.append((
                tag, m.get("model", "-"), f"{m.get('parameters', 0):,}",
                pct(m["accuracy"]), pct(m["balanced_accuracy"]),
                pct(m["macro_f1"]),
            ))
        elif "results" in blob:
            r = blob["results"]
            seen = r.get("seen_classes", {})
            ho = r.get("held_out_classes", {})
            fmt = lambda d: (f"{pct(d['accuracy'])} +- {pct(d['ci95'])}"
                             if "accuracy" in d else "n/a")
            maml_runs.append((
                tag, f"{r.get('n_way')}-way {r.get('k_shot')}-shot",
                r.get("inner_steps"), "FO" if r.get("first_order") else "2nd",
                fmt(seen), fmt(ho),
            ))

    lines = ["# Experimental results", ""]
    if baselines:
        lines += ["## Supervised baselines (full 7-class test set)", "",
                  markdown_table(baselines, ["run", "model", "params",
                                             "accuracy %", "balanced acc %",
                                             "macro-F1 %"]), ""]
    if maml_runs:
        lines += ["## MAML (episodic, mean over episodes with 95% CI)", "",
                  markdown_table(maml_runs, ["run", "setting", "inner steps",
                                             "order", "seen acc %",
                                             "held-out acc %"]), ""]
    if not baselines and not maml_runs:
        lines += ["_No completed runs found._", ""]

    out = results_dir / "REPORT.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
