"""ImageNet-pretrained backbone fine-tuned on FER2013.

This is the standard recipe behind published FER2013 results in the high 60s /
low 70s, and the single largest jump over a from-scratch CNN. The pieces that
matter, in order of contribution:

  1. ImageNet-pretrained ResNet, fine-tuned end to end.
  2. Upscaling 48x48 -> 224x224 so the pretrained filters see structure at the
     scale they were trained for. Feeding 48x48 directly wastes the weights.
  3. Grayscale replicated to 3 channels, with the first conv's weights summed
     appropriately so the pretrained stem still works.
  4. Test-time augmentation (horizontal flip) at evaluation.

Checkpoints are saved so several runs can be ensembled by predict_ensemble.py.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from config import get_config, NUM_CLASSES
from data import build_dataloaders, class_weights
from evaluate import (evaluate_supervised, per_class_table,
                      save_confusion_matrix, save_curves, save_json)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_pretrained(arch="resnet18", num_classes=NUM_CLASSES, dropout=0.2):
    """Load an ImageNet ResNet and adapt it to 1-channel input."""
    ctor = getattr(torchvision.models, arch)
    model = ctor(weights=None)

    # Weights come from the local torch hub cache; no network access needed.
    ckpt = {
        "resnet18": "resnet18-f37072fd.pth",
        "resnet34": "resnet34-b627a593.pth",
    }[arch]
    path = Path.home() / ".cache/torch/hub/checkpoints" / ckpt
    if not path.exists():
        raise SystemExit(f"pretrained weights not found at {path}")
    model.load_state_dict(torch.load(path, map_location="cpu"))

    # Collapse the RGB stem to a single channel by summing across the input
    # dimension. This preserves the filters' response to grayscale structure,
    # which is what replicating the channel three times would also achieve but
    # at a third of the compute.
    w = model.conv1.weight.data
    stem = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    stem.weight.data = w.sum(dim=1, keepdim=True)
    model.conv1 = stem

    in_feat = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
    return model


class Upscale(nn.Module):
    """Resize a batch on the GPU, so the dataloader still moves 48x48 tensors."""

    def __init__(self, size, model):
        super().__init__()
        self.size, self.model = size, model

    def forward(self, x):
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size),
                              mode="bilinear", align_corners=False)
        return self.model(x)


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
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


@torch.no_grad()
def evaluate_tta(model, loader, device):
    """Average logits over the image and its horizontal flip."""
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).softmax(1) + model(torch.flip(x, dims=[3])).softmax(1)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    y_pred, y_true = np.concatenate(preds), np.concatenate(trues)
    from sklearn.metrics import balanced_accuracy_score, f1_score
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "y_pred": y_pred, "y_true": y_true,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--monitor", default="val_loss",
                    choices=["val_loss", "val_acc"],
                    help="metric for checkpointing and early stopping. "
                         "Declared before the run, never chosen after seeing "
                         "the curve.")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    tag = args.tag or f"pretrained_{args.arch}_s{args.seed}"
    out_dir = Path(cfg.results_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] device={device} arch={args.arch} img={args.img_size}")

    train_loader, val_loader, test_loader, train_labels = build_dataloaders(
        cfg, batch_size=args.batch_size, aug_strength=2)

    if args.limit_batches:
        from itertools import islice

        class _Limited:
            def __init__(self, dl, n): self.dl, self.n = dl, n
            def __iter__(self): return islice(iter(self.dl), self.n)
            def __len__(self): return min(self.n, len(self.dl))

        train_loader = _Limited(train_loader, args.limit_batches)
        val_loader = _Limited(val_loader, args.limit_batches)
        test_loader = _Limited(test_loader, args.limit_batches)

    model = Upscale(args.img_size, build_pretrained(args.arch)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] parameters: {n_params:,}")

    use_w = cfg.baseline.class_weighted_loss and not args.no_class_weights
    weight = class_weights(train_labels, NUM_CLASSES).to(device) if use_w else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # Under label smoothing on a noisy-label dataset, validation loss can rise
    # while accuracy keeps improving, so the two criteria disagree. Which one
    # governs is a declared choice, fixed before training starts.
    track_acc = args.monitor == "val_acc"
    best_val, best_epoch, bad = (-float("inf") if track_acc else float("inf")), -1, 0
    ckpt = out_dir / "model.pt"
    t0 = time.time()

    for ep in range(args.epochs):
        tl, ta = run_epoch(model, train_loader, device, criterion, optimizer)
        vl, va = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        for k, v in zip(history, (tl, ta, vl, va)):
            history[k].append(v)
        print(f"[{tag}] epoch {ep+1}/{args.epochs} "
              f"train_loss={tl:.4f} train_acc={ta:.4f} "
              f"val_loss={vl:.4f} val_acc={va:.4f}", flush=True)

        current = va if track_acc else vl
        improved = current > best_val if track_acc else current < best_val
        if improved:
            best_val, best_epoch, bad = current, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{tag}] early stop at epoch {ep+1}")
                break

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))

    plain = evaluate_supervised(model, test_loader, device)
    tta = evaluate_tta(model, test_loader, device)
    y_pred, y_true = tta.pop("y_pred"), tta.pop("y_true")
    plain.pop("y_pred"); plain.pop("y_true")

    metrics = {
        "arch": args.arch, "tag": tag, "seed": args.seed,
        "img_size": args.img_size, "parameters": n_params,
        "epochs_run": len(history["train_loss"]), "best_epoch": best_epoch + 1,
        "monitor": args.monitor, "patience": args.patience,
        "train_minutes": (time.time() - t0) / 60,
        "plain": {k: plain[k] for k in
                  ("accuracy", "balanced_accuracy", "macro_f1")},
        "tta": tta,
    }
    # The report reads these top-level keys; TTA is the headline number.
    metrics.update(accuracy=tta["accuracy"],
                   balanced_accuracy=tta["balanced_accuracy"],
                   macro_f1=tta["macro_f1"])

    save_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png",
                          title=f"{args.arch} +TTA - FER2013 test")
    save_curves(history, out_dir / "curves.png", title=tag)
    save_json({"metrics": metrics,
               "per_class": per_class_table(y_true, y_pred),
               "history": history}, out_dir / "metrics.json")

    print(f"[{tag}] plain acc={plain['accuracy']:.4f} | "
          f"TTA acc={tta['accuracy']:.4f} "
          f"balanced={tta['balanced_accuracy']:.4f} "
          f"macro_f1={tta['macro_f1']:.4f}")
    print(f"[{tag}] wrote {out_dir}")


if __name__ == "__main__":
    main()
