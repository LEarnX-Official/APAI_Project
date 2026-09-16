"""Ensemble several fine-tuned checkpoints, with test-time augmentation.

Averaging softmax outputs over independently-seeded runs is the cheapest
remaining gain once a single model has converged: the errors of separate runs
are partly uncorrelated, so the mean is better than any member. Every strong
FER2013 entry does this.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import get_config, NUM_CLASSES
from data import build_dataloaders
from evaluate import per_class_table, save_confusion_matrix, save_json
from train_pretrained import Upscale, build_pretrained


@torch.no_grad()
def member_probs(model, loader, device, tta=True):
    """Softmax probabilities for one model over the whole loader."""
    model.eval()
    out, trues = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        p = model(x).softmax(1)
        if tta:
            p = p + model(torch.flip(x, dims=[3])).softmax(1)
            p = p / 2
        out.append(p.cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(out), np.concatenate(trues)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="result tags to ensemble, e.g. pretrained_resnet18_s12")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--tag", default="ensemble")
    args = ap.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(cfg.results_dir)
    out_dir = results_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader, _ = build_dataloaders(cfg, batch_size=args.batch_size)

    probs, y_true, members = None, None, []
    for tag in args.runs:
        run_dir = results_dir / tag
        ckpt, meta = run_dir / "model.pt", run_dir / "metrics.json"
        if not ckpt.exists():
            print(f"[{args.tag}] skip {tag}: no checkpoint")
            continue
        arch, img = "resnet18", 224
        if meta.exists():
            m = json.loads(meta.read_text()).get("metrics", {})
            arch, img = m.get("arch", arch), m.get("img_size", img)

        model = Upscale(img, build_pretrained(arch, NUM_CLASSES)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        p, t = member_probs(model, test_loader, device, tta=not args.no_tta)
        acc = float((p.argmax(1) == t).mean())
        print(f"[{args.tag}] member {tag} ({arch}): {100*acc:.2f}%")
        members.append({"tag": tag, "arch": arch, "accuracy": acc})

        probs = p if probs is None else probs + p
        y_true = t
        del model
        torch.cuda.empty_cache()

    if not members:
        raise SystemExit("no usable checkpoints given")

    y_pred = (probs / len(members)).argmax(1)
    from sklearn.metrics import balanced_accuracy_score, f1_score
    metrics = {
        "tag": args.tag,
        "n_members": len(members),
        "members": members,
        "tta": not args.no_tta,
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    save_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png",
                          title=f"Ensemble of {len(members)} - FER2013 test")
    save_json({"metrics": metrics,
               "per_class": per_class_table(y_true, y_pred)},
              out_dir / "metrics.json")

    best = max(m["accuracy"] for m in members)
    print(f"[{args.tag}] best single member: {100*best:.2f}%")
    print(f"[{args.tag}] ENSEMBLE acc={100*metrics['accuracy']:.2f}% "
          f"balanced={100*metrics['balanced_accuracy']:.2f}% "
          f"macro_f1={100*metrics['macro_f1']:.2f}%")
    print(f"[{args.tag}] wrote {out_dir}")


if __name__ == "__main__":
    main()
