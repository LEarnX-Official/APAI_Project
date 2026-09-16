# Experimental results

## Supervised baselines (full 7-class test set)

| run | model | params | accuracy % | balanced acc % | macro-F1 % |
|---|---|---|---|---|---|
| baseline_convnet4 | convnet4 | 115,975 | 49.61 | 50.83 | 46.38 |
| baseline_simple_cnn | simple_cnn | 1,274,823 | 42.89 | 37.20 | 33.98 |
| convnet4_tuned | convnet4 | 453,127 | 50.93 | 52.11 | 47.53 |
| pretrained_resnet18_acc | - | 11,173,831 | 69.23 | 68.18 | 66.29 |
| pretrained_resnet18_s12 | - | 11,173,831 | 66.93 | 65.57 | 62.48 |

## MAML (episodic, mean over episodes with 95% CI)

| run | setting | inner steps | order | seen acc % | held-out acc % |
|---|---|---|---|---|---|
| maml_5way_5shot | 5-way 5-shot | 5 | FO | 39.41 +- 0.67 | 35.63 +- 0.64 |
