"""Central configuration for all experiments.

Every hyperparameter the paper reports lives here so that a run is fully
described by this file plus a git commit.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

# FER2013 class names, in the order used for every label index in this project.
CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
NUM_CLASSES = len(CLASSES)

# Human-readable names for figures and tables in the paper.
DISPLAY_NAMES = {
    "angry": "Anger",
    "disgusted": "Disgust",
    "fearful": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sadness",
    "surprised": "Surprise",
}

IMG_SIZE = 48


@dataclass
class DataConfig:
    root: Path = DATA_ROOT
    img_size: int = IMG_SIZE
    val_split: float = 0.2
    seed: int = 12
    # Per-channel statistics computed over the training split; see
    # scripts/compute_stats.py. Grayscale, so a single value each.
    mean: float = 0.5077
    std: float = 0.2550


@dataclass
class BaselineConfig:
    """Supervised baselines: the comparison points the APAI brief requires.

    Defaults are the *tuned* settings. The first 30-epoch runs stopped while
    still improving, so the schedule is the binding constraint, not the model.
    """
    epochs: int = 150
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Counteracts the 16:1 happy-to-disgust imbalance in FER2013.
    class_weighted_loss: bool = True
    # Generous, because cosine annealing dips before it recovers; a tight
    # patience kills the run during a scheduled trough.
    early_stopping_patience: int = 25
    # 1 = flip + shift, 2 = adds rotation, scaling, random erasing.
    aug_strength: int = 2
    # "cosine" anneals over the full run; "plateau" is the earlier behaviour.
    scheduler: str = "cosine"
    # Channel width for convnet4; 64 was the first configuration.
    width: int = 128
    label_smoothing: float = 0.05


@dataclass
class MAMLConfig:
    """First-order MAML for few-shot emotion recognition."""
    n_way: int = 5            # classes sampled per episode
    k_shot: int = 5           # support examples per class
    q_query: int = 15         # query examples per class
    inner_steps: int = 5
    inner_lr: float = 0.01
    meta_lr: float = 0.001
    meta_batch_size: int = 4  # tasks aggregated per meta-update
    meta_iterations: int = 2000
    first_order: bool = True  # FOMAML; set False for full second-order
    eval_episodes: int = 300
    # Classes held out entirely from meta-training, to test generalisation to
    # emotions never seen during training. Chosen as the two rarest.
    held_out_classes: tuple = ("disgusted", "surprised")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    maml: MAMLConfig = field(default_factory=MAMLConfig)
    device: str = "cuda"
    seed: int = 12
    results_dir: Path = RESULTS_ROOT

    def save(self, path):
        def _enc(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            return o
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=_enc))


def get_config() -> Config:
    return Config()
