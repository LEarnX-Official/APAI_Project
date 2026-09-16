# Few-Shot Facial Emotion Recognition with MAML

APAI (AA. 2024-2025) course project. Meta-learning for facial expression
recognition on FER2013, with supervised baselines for comparison.

## Motivation

FER2013 is severely imbalanced: 7,215 `happy` images against 436 `disgusted`,
a 16:1 ratio. Conventional supervised training handles the frequent classes and
largely ignores the rare ones. This project asks whether MAML can recognise
emotions from only a handful of labelled examples, including emotions withheld
entirely from meta-training.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch. A CUDA GPU is optional but strongly
recommended for the full runs.

## Data

**The dataset is not included in this repository.** FER2013 is a public Kaggle
dataset and is not redistributed here; download it and lay it out as:

```
data/{train,test}/{angry,disgusted,fearful,happy,neutral,sad,surprised}/*.png
```

Source: <https://www.kaggle.com/datasets/msambare/fer2013> (48x48 grayscale
PNGs). Some mirrors use different class spellings -- the directory names above
are what `src/config.py` expects, so rename if yours differ.

Expected once in place: 28,709 training and 7,178 test images, a 16.5:1
imbalance between `happy` (7,215) and `disgusted` (436). Verify with:

```bash
python scripts/inspect_data.py
```

## Running

Verify the pipeline end to end (about a minute, tiny settings):

```bash
bash scripts/smoke_test.sh
```

Full experiments:

```bash
# Baseline 1: supervised CNN
python src/train_baseline.py --model simple_cnn

# Baseline 2: supervised ConvNet4, the MAML backbone trained conventionally,
# which isolates the contribution of meta-learning from the architecture
python src/train_baseline.py --model convnet4

# Proposed: first-order MAML, 5-way 5-shot
python src/train_maml.py

# Ablations
python src/train_maml.py --k-shot 1  --tag maml_5way_1shot
python src/train_maml.py --inner-steps 1 --tag maml_inner1
python src/train_maml.py --second-order  --tag maml_second_order
```

Every run writes to `results/<tag>/`: `metrics.json`, `confusion_matrix*.png`,
`curves.png`, and the best checkpoint.

Collect all finished runs into paper-ready tables:

```bash
python scripts/make_report.py
```

## Layout

```
src/config.py          all hyperparameters
src/data.py            FER2013 loading, episodic sampler
src/models.py          SimpleCNN baseline, ConvNet4 with functional forward
src/maml.py            inner loop, meta-update, meta-evaluation
src/train_baseline.py  supervised training entry point
src/train_maml.py      meta-training entry point
src/evaluate.py        metrics and figures
scripts/               smoke test, data inspection, report generation
paper/                 CVPR-format report draft
results/               experiment outputs
```

## Method

**Episodes.** An N-way K-shot task samples N emotions, K support and Q query
images each, relabelled 0..N-1 within the episode.

**Inner loop.** From initialisation theta, K support examples drive `inner_steps`
of gradient descent at `inner_lr`, giving adapted parameters theta'.

**Outer loop.** The query loss at theta' is differentiated with respect to the
original theta, so the initialisation is optimised for adaptability rather than
for immediate accuracy. First-order MAML (the default) drops the second
derivative for tractability; `--second-order` enables the full computation.

**Held-out protocol.** `disgusted` and `surprised` are excluded from
meta-training entirely and used only at meta-test, which tests generalisation
to genuinely unseen emotions rather than to unseen images of known emotions.

## Notes on evaluation

Accuracy alone is misleading under this imbalance, so balanced accuracy and
macro-F1 are reported alongside it. Few-shot numbers are averaged over episodes
and reported with a 95% confidence interval, as is standard in the literature.
