"""Model architectures.

Two backbones:
  * SimpleCNN  - the supervised baseline, deliberately close to the CNN used in
                 the original project code so the comparison is fair.
  * ConvNet4   - the 4-block convolutional encoder standard in the MAML and
                 few-shot literature (Vinyals et al. 2016; Finn et al. 2017).

ConvNet4 additionally supports a *functional* forward pass, taking an explicit
parameter dict. That is what lets the MAML inner loop compute updated weights
without mutating the module in place.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, IMG_SIZE


class SimpleCNN(nn.Module):
    """Supervised baseline: 3 conv blocks + classifier head."""

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) ** 2, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _conv_block_params(in_c, out_c):
    conv = nn.Conv2d(in_c, out_c, 3, padding=1)
    bn = nn.BatchNorm2d(out_c)
    return conv, bn


class ConvNet4(nn.Module):
    """4-block ConvNet with an optional functional forward.

    Uses track_running_stats=False in BatchNorm: with episodic training the
    batch statistics differ per task, so running averages accumulated across
    tasks would leak information between episodes and hurt adaptation. This is
    the standard choice in few-shot work.
    """

    def __init__(self, num_classes=NUM_CLASSES, hidden=64):
        super().__init__()
        chans = [1, hidden, hidden, hidden, hidden]
        for i in range(4):
            conv, bn = _conv_block_params(chans[i], chans[i + 1])
            setattr(self, f"conv{i}", conv)
            setattr(self, f"bn{i}", nn.BatchNorm2d(
                chans[i + 1], track_running_stats=False))
        # 48 -> 24 -> 12 -> 6 -> 3
        self.feat_dim = hidden * (IMG_SIZE // 16) ** 2
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, params=None):
        if params is None:
            for i in range(4):
                x = getattr(self, f"conv{i}")(x)
                x = getattr(self, f"bn{i}")(x)
                x = F.max_pool2d(F.relu(x), 2)
            return self.head(x.flatten(1))
        return self.functional_forward(x, params)

    def functional_forward(self, x, params):
        """Forward pass using externally supplied weights.

        `params` maps this module's parameter names to tensors. Because the
        tensors are used directly rather than assigned to the module, autograd
        can track the inner-loop updates for a second-order meta-gradient.
        """
        for i in range(4):
            x = F.conv2d(x, params[f"conv{i}.weight"], params[f"conv{i}.bias"],
                         padding=1)
            x = F.batch_norm(
                x, None, None,
                params.get(f"bn{i}.weight"), params.get(f"bn{i}.bias"),
                training=True,
            )
            x = F.max_pool2d(F.relu(x), 2)
        return F.linear(x.flatten(1), params["head.weight"], params["head.bias"])

    def clone_params(self):
        return OrderedDict((n, p.clone()) for n, p in self.named_parameters())


def build_model(name, num_classes=NUM_CLASSES, width=64):
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    if name == "convnet4":
        return ConvNet4(num_classes, hidden=width)
    raise ValueError(f"unknown model '{name}' (expected simple_cnn|convnet4)")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
