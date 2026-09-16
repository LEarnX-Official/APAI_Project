"""Model-Agnostic Meta-Learning (Finn et al., 2017) for few-shot FER.

The inner loop adapts a copy of the parameters to an episode's support set by
plain gradient descent; the outer loop updates the initialisation so that a few
such steps generalise to the query set.

The key correctness point, and the thing the original project code got wrong:
the meta-gradient must be taken of the *post-adaptation query loss* with respect
to the *initial* parameters. Differencing weight vectors and dividing by the
learning rate is not that quantity.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F


def inner_adapt(model, params, x, y, inner_lr, steps, first_order):
    """Run the inner loop; return adapted parameters.

    With first_order=True the graph through the inner updates is dropped
    (FOMAML), which costs a little accuracy and saves a lot of memory.
    """
    for _ in range(steps):
        logits = model.functional_forward(x, params)
        loss = F.cross_entropy(logits, y)
        grads = torch.autograd.grad(
            loss, list(params.values()),
            create_graph=not first_order,
            allow_unused=True,
        )
        params = OrderedDict(
            (name, p if g is None else p - inner_lr * g)
            for (name, p), g in zip(params.items(), grads)
        )
    return params


def meta_train_step(model, sampler, cfg, optimizer, device):
    """One meta-update aggregated over `meta_batch_size` episodes."""
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_acc = 0.0, 0.0

    for _ in range(cfg.meta_batch_size):
        sx, sy, qx, qy = sampler.sample(device)
        adapted = inner_adapt(
            model, model.clone_params(), sx, sy,
            cfg.inner_lr, cfg.inner_steps, cfg.first_order,
        )
        q_logits = model.functional_forward(qx, adapted)
        q_loss = F.cross_entropy(q_logits, qy) / cfg.meta_batch_size
        q_loss.backward()

        total_loss += q_loss.item() * cfg.meta_batch_size
        total_acc += (q_logits.argmax(1) == qy).float().mean().item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    n = cfg.meta_batch_size
    return total_loss / n, total_acc / n


@torch.no_grad()
def _accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def meta_evaluate(model, sampler, cfg, device, episodes=None, collect=False):
    """Meta-test: adapt to each episode's support set, score on its query set.

    Returns mean accuracy, the 95% confidence interval, and optionally the raw
    per-episode predictions for a confusion matrix.
    """
    n_episodes = episodes or cfg.eval_episodes
    accs, preds, trues = [], [], []

    for _ in range(n_episodes):
        sx, sy, qx, qy = sampler.sample(device)
        # Adaptation needs gradients even at eval time.
        with torch.enable_grad():
            adapted = inner_adapt(
                model, model.clone_params(), sx, sy,
                cfg.inner_lr, cfg.inner_steps, first_order=True,
            )
        with torch.no_grad():
            logits = model.functional_forward(qx, adapted)
        accs.append(_accuracy(logits, qy))
        if collect:
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(qy.cpu().numpy())

    accs = np.asarray(accs)
    mean = float(accs.mean())
    ci95 = float(1.96 * accs.std(ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else 0.0

    out = {"accuracy": mean, "ci95": ci95, "n_episodes": n_episodes}
    if collect:
        out["y_pred"] = np.concatenate(preds)
        out["y_true"] = np.concatenate(trues)
    return out
