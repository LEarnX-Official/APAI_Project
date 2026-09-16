"""Meta-train MAML for few-shot facial emotion recognition.

Two evaluation protocols:
  * seen     - episodes drawn from the classes used during meta-training.
  * held-out - episodes drawn from emotions never seen in meta-training,
               which is the actual few-shot generalisation claim.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from config import get_config, CLASSES
from data import load_split_array, stratified_split, EpisodeSampler
from models import build_model, count_parameters
from maml import meta_train_step, meta_evaluate
from evaluate import save_confusion_matrix, save_curves, save_json


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--n-way", type=int, default=None)
    ap.add_argument("--k-shot", type=int, default=None)
    ap.add_argument("--inner-steps", type=int, default=None)
    ap.add_argument("--inner-lr", type=float, default=None)
    ap.add_argument("--meta-lr", type=float, default=None)
    ap.add_argument("--second-order", action="store_true",
                    help="full MAML instead of the first-order approximation")
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = get_config()
    m = cfg.maml
    for src, dst in (("iterations", "meta_iterations"), ("n_way", "n_way"),
                     ("k_shot", "k_shot"), ("inner_steps", "inner_steps"),
                     ("inner_lr", "inner_lr"), ("meta_lr", "meta_lr"),
                     ("eval_episodes", "eval_episodes")):
        v = getattr(args, src)
        if v is not None:
            setattr(m, dst, v)
    if args.second_order:
        m.first_order = False

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    tag = args.tag or f"maml_{m.n_way}way_{m.k_shot}shot"
    out_dir = Path(cfg.results_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] device={device} first_order={m.first_order}")

    images, labels = load_split_array(cfg.data.root, "train")
    tr_idx, va_idx = stratified_split(labels, cfg.data.val_split, cfg.data.seed)

    held_out = [CLASSES.index(c) for c in m.held_out_classes]
    seen = [i for i in range(len(CLASSES)) if i not in held_out]
    if m.n_way > len(seen):
        raise SystemExit(
            f"n_way={m.n_way} exceeds the {len(seen)} meta-training classes "
            f"left after holding out {list(m.held_out_classes)}"
        )
    print(f"[{tag}] meta-train classes: {[CLASSES[i] for i in seen]}")
    print(f"[{tag}] held-out classes:   {[CLASSES[i] for i in held_out]}")

    mk = lambda idx, classes, seed: EpisodeSampler(
        images[idx], labels[idx], classes, m.n_way, m.k_shot, m.q_query,
        cfg.data.mean, cfg.data.std, seed=seed)

    train_sampler = mk(tr_idx, seen, cfg.seed)
    val_sampler = mk(va_idx, seen, cfg.seed + 1)
    # Held-out episodes need n_way classes; fall back to all held-out classes
    # when fewer than n_way were reserved.
    ho_classes = held_out if len(held_out) >= m.n_way else held_out + seen[:m.n_way - len(held_out)]

    model = build_model("convnet4", num_classes=m.n_way).to(device)
    print(f"[{tag}] parameters: {count_parameters(model):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=m.meta_lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, ckpt = -1.0, out_dir / "model.pt"
    running = []
    t0 = time.time()

    for it in range(1, m.meta_iterations + 1):
        loss, acc = meta_train_step(model, train_sampler, m, optimizer, device)
        running.append((loss, acc))

        if it % args.eval_every == 0 or it == m.meta_iterations:
            tr_loss = float(np.mean([r[0] for r in running]))
            tr_acc = float(np.mean([r[1] for r in running]))
            running = []
            val = meta_evaluate(model, val_sampler, m, device,
                                episodes=max(50, m.eval_episodes // 4))
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(float("nan"))
            history["val_acc"].append(val["accuracy"])
            print(f"[{tag}] iter {it}/{m.meta_iterations} "
                  f"meta_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                  f"val_acc={val['accuracy']:.4f}+-{val['ci95']:.4f}")
            if val["accuracy"] > best_acc:
                best_acc = val["accuracy"]
                torch.save(model.state_dict(), ckpt)

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))

    results = {"tag": tag, "parameters": count_parameters(model),
               "first_order": m.first_order, "n_way": m.n_way,
               "k_shot": m.k_shot, "inner_steps": m.inner_steps,
               "inner_lr": m.inner_lr, "meta_lr": m.meta_lr,
               "iterations": m.meta_iterations,
               "train_minutes": (time.time() - t0) / 60}

    seen_res = meta_evaluate(model, val_sampler, m, device, collect=True)
    y_pred, y_true = seen_res.pop("y_pred"), seen_res.pop("y_true")
    results["seen_classes"] = seen_res
    ep_names = [f"class {i}" for i in range(m.n_way)]
    save_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix_seen.png",
                          labels=ep_names, title=f"{tag} - seen classes")
    print(f"[{tag}] seen  acc={seen_res['accuracy']:.4f}+-{seen_res['ci95']:.4f}")

    try:
        ho_sampler = mk(va_idx, ho_classes, cfg.seed + 2)
        ho_res = meta_evaluate(model, ho_sampler, m, device, collect=True)
        yp, yt = ho_res.pop("y_pred"), ho_res.pop("y_true")
        results["held_out_classes"] = ho_res
        save_confusion_matrix(yt, yp, out_dir / "confusion_matrix_heldout.png",
                              labels=ep_names, title=f"{tag} - held-out classes")
        print(f"[{tag}] held-out acc={ho_res['accuracy']:.4f}+-{ho_res['ci95']:.4f}")
    except ValueError as e:
        results["held_out_classes"] = {"error": str(e)}
        print(f"[{tag}] held-out evaluation skipped: {e}")

    save_curves(history, out_dir / "curves.png", title=tag)
    save_json({"results": results, "history": history}, out_dir / "metrics.json")
    print(f"[{tag}] wrote {out_dir}")


if __name__ == "__main__":
    main()
