"""Train the supervised baselines.

These are the comparison points the APAI brief explicitly requires: a naive
method to justify the effectiveness of the proposed approach.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import get_config, CLASSES, NUM_CLASSES
from data import build_dataloaders, class_weights
from models import build_model, count_parameters
from evaluate import (evaluate_supervised, per_class_table, save_confusion_matrix,
                      save_curves, save_json)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()

    return loss_sum / max(n, 1), correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="simple_cnn",
                    choices=["simple_cnn", "convnet4"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="debug: stop each epoch after N batches")
    ap.add_argument("--width", type=int, default=None,
                    help="convnet4 channel width")
    ap.add_argument("--aug-strength", type=int, default=None, choices=[1, 2],
                    help="1=flip+shift, 2=+rotation/scale/erasing")
    ap.add_argument("--scheduler", default=None, choices=["cosine", "plateau"])
    ap.add_argument("--label-smoothing", type=float, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = get_config()
    epochs = args.epochs or cfg.baseline.epochs
    lr = args.lr or cfg.baseline.lr
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    tag = args.tag or f"baseline_{args.model}"
    out_dir = Path(cfg.results_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{tag}] device={device}")
    train_loader, val_loader, test_loader, train_labels = build_dataloaders(
        cfg, batch_size=args.batch_size, aug_strength=args.aug_strength)

    if args.limit_batches:
        from itertools import islice

        class _Limited:
            def __init__(self, dl, n): self.dl, self.n = dl, n
            def __iter__(self): return islice(iter(self.dl), self.n)
            def __len__(self): return min(self.n, len(self.dl))

        train_loader = _Limited(train_loader, args.limit_batches)
        val_loader = _Limited(val_loader, args.limit_batches)
        test_loader = _Limited(test_loader, args.limit_batches)

    width = args.width if args.width is not None else cfg.baseline.width
    model = build_model(args.model, width=width).to(device)
    print(f"[{tag}] parameters: {count_parameters(model):,}")

    use_w = cfg.baseline.class_weighted_loss and not args.no_class_weights
    weight = class_weights(train_labels, NUM_CLASSES).to(device) if use_w else None
    smoothing = (args.label_smoothing if args.label_smoothing is not None
                 else cfg.baseline.label_smoothing)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg.baseline.weight_decay)

    sched_name = args.scheduler or cfg.baseline.scheduler
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_epoch, bad = float("inf"), -1, 0
    ckpt = out_dir / "model.pt"
    t0 = time.time()

    for ep in range(epochs):
        tl, ta = run_epoch(model, train_loader, device, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, device, criterion)
        # Cosine steps on epoch count; plateau needs the metric it monitors.
        scheduler.step() if sched_name == "cosine" else scheduler.step(vl)
        for k, v in zip(history, (tl, ta, vl, va)):
            history[k].append(v)
        print(f"[{tag}] epoch {ep+1}/{epochs} "
              f"train_loss={tl:.4f} train_acc={ta:.4f} "
              f"val_loss={vl:.4f} val_acc={va:.4f}")

        if vl < best_val:
            best_val, best_epoch, bad = vl, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= cfg.baseline.early_stopping_patience:
                print(f"[{tag}] early stop at epoch {ep+1}")
                break

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))

    metrics = evaluate_supervised(model, test_loader, device)
    y_pred, y_true = metrics.pop("y_pred"), metrics.pop("y_true")
    metrics.update(
        model=args.model, tag=tag, epochs_run=len(history["train_loss"]),
        best_epoch=best_epoch + 1, parameters=count_parameters(model),
        class_weighted=bool(use_w), train_minutes=(time.time() - t0) / 60,
    )

    save_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png",
                          title=f"{args.model} - FER2013 test")
    save_curves(history, out_dir / "curves.png", title=f"{args.model}")
    save_json({"metrics": metrics,
               "per_class": per_class_table(y_true, y_pred),
               "history": history}, out_dir / "metrics.json")

    print(f"[{tag}] test acc={metrics['accuracy']:.4f} "
          f"balanced={metrics['balanced_accuracy']:.4f} "
          f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"[{tag}] wrote {out_dir}")


if __name__ == "__main__":
    main()
