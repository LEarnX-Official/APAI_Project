# Paper draft — what is done and what you must do

`main.tex` is a complete draft covering all six mandatory APAI sections, with
every number taken from `results/*/metrics.json`. It is **not submittable as
is**. The items below are yours.

## Blocking — must be done before submission

1. **Swap in the real CVPR 2025 style files.**
   Download from https://github.com/cvpr-org/author-kit/releases, put
   `cvpr.sty` (and companions) beside `main.tex`, and replace the
   `\usepackage[...]{geometry}` block at the top with `\usepackage{cvpr}`.
   The geometry currently there only approximates the layout so the file
   compiles standalone.

2. **Verify the page count against the real template.**
   6 pages minimum, 8 maximum, excluding references. *"A paper not obeying the
   page limits will be firmly rejected."* The current draft is close to the
   lower bound; with the real style file it will reflow.

3. **Fill in the Contributions section.** Placeholders are marked `TODO`.
   The brief warns that undisclosed under-contribution can affect the whole
   team's grade. Be accurate rather than even-handed.

4. **Replace the author block** with real names and addresses.

5. **Publish the repository and put its URL in the abstract.**
   `\url{https://github.com/USERNAME/REPOSITORY}` is a placeholder. The repo
   must be public — the brief requires the code link in the abstract.

## Recommended

6. **Add figures from `results/`.** The draft is text-and-tables only. Good
   candidates, all already generated:
   - `results/pretrained_resnet18_acc/confusion_matrix.png`
   - `results/pretrained_resnet18_acc/curves.png` (shows the loss/accuracy
     divergence that motivates the checkpoint ablation)
   - `results/maml_5way_5shot/confusion_matrix_heldout.png`

   Insert with `\includegraphics[width=\linewidth]{...}` inside a `figure`
   environment. Each one you add costs space against the page limit.

7. **Read two or three CVPR papers** before finalising, as the brief suggests,
   and match their tone in the Introduction.

## Notes on choices made in the draft

- **The pipeline diagram is drawn in TikZ**, inline in `main.tex`
  (Fig. 1). The brief marks this `[VERY IMPORTANT]`, so it is a real figure
  rather than a placeholder. Edit the node text directly if you change the
  method.

- **The few-shot and supervised numbers are kept in separate tables**, with an
  explicit paragraph explaining why. They are 5-way (20% chance) and 7-way
  (14.3% chance) respectively and are not comparable. Merging them would
  inflate the apparent result and a reviewer would catch it.

- **The negative result is reported**, not hidden: 150-epoch ConvNet4 training
  bought only +1.32 points for 8× the compute. This is what identified
  architecture as the bottleneck, and it reads as honest rather than weak.

- **The `np.roll` augmentation defect** is written up in the Method section.
  It was a real bug in our own first implementation.

- **Limitations are stated plainly**, including single-seed runs, the two-class
  held-out set, and the 4–7 point gap to published SOTA. Reviewers find these
  anyway; stating them first is stronger.

## Compiling

```bash
cd paper
pdflatex main.tex && pdflatex main.tex   # twice, for references
```

TikZ is required (standard in TeX Live / MiKTeX full installs).
