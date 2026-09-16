"""Metrics, tables, and figures for the Results section."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, f1_score)

from config import CLASSES, DISPLAY_NAMES


@torch.no_grad()
def evaluate_supervised(model, loader, device, num_classes=len(CLASSES)):
    """Accuracy, macro-F1, balanced accuracy, and predictions for a baseline."""
    model.eval()
    preds, trues, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        n += y.numel()
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.cpu().numpy())

    y_pred, y_true = np.concatenate(preds), np.concatenate(trues)
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": float((y_pred == y_true).mean()),
        # Balanced accuracy matters here: FER2013 is 16:1 imbalanced, so plain
        # accuracy flatters a model that ignores the rare classes.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "y_pred": y_pred,
        "y_true": y_true,
    }


def per_class_table(y_true, y_pred, labels=None):
    names = labels or CLASSES
    rep = classification_report(
        y_true, y_pred, labels=list(range(len(names))),
        target_names=names, output_dict=True, zero_division=0,
    )
    return rep


def save_confusion_matrix(y_true, y_pred, path, labels=None, normalize=True,
                          title="Confusion Matrix"):
    """Write a confusion-matrix figure. Row-normalised by default, which is the
    informative view under heavy class imbalance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = labels or [DISPLAY_NAMES.get(c, c) for c in CLASSES]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    if normalize:
        with np.errstate(all="ignore"):
            cm_disp = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
            cm_disp = np.nan_to_num(cm_disp)
    else:
        cm_disp = cm

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm_disp, cmap="Blues", vmin=0,
                   vmax=1 if normalize else cm_disp.max())
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)

    thresh = (1 if normalize else cm_disp.max()) / 2
    for i in range(len(names)):
        for j in range(len(names)):
            txt = f"{cm_disp[i, j]:.2f}" if normalize else f"{int(cm_disp[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if cm_disp[i, j] > thresh else "black")

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return cm


def save_curves(history, path, title="Training curves"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax, name in (("loss", axes[0], "Loss"), ("acc", axes[1], "Accuracy")):
        for split in ("train", "val"):
            k = f"{split}_{key}"
            if history.get(k):
                ax.plot(history[k], label=split, marker="o", markersize=3)
        ax.set_xlabel("Epoch"); ax.set_ylabel(name); ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_json(obj, path):
    def _enc(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return o
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_enc))


def markdown_table(rows, headers):
    """Emit a markdown table, ready to paste into the paper draft."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
