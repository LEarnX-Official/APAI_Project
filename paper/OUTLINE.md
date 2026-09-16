# Paper outline — APAI requirements checklist

Target: 6–8 pages excluding references, CVPR 2025 template
(https://github.com/cvpr-org/author-kit/releases). Papers outside the page
limit are rejected outright.

**The code URL goes in the abstract, and the repository must be public.**

---

## 1. Abstract
- [ ] Problem and why it matters
- [ ] Key challenges (imbalance, rare emotions, few labels)
- [ ] The proposed approach in two sentences
- [ ] Headline results
- [ ] Impact statement
- [ ] **Public code URL**

## 2. Introduction
- [ ] Define the problem: facial emotion recognition from static images
- [ ] Significance + real-world applications (HCI, driver monitoring,
      assistive tech, clinical screening)
- [ ] Challenges: 16:1 class imbalance, low 48×48 resolution, label noise in
      FER2013, expensive annotation for rare emotions
- [ ] Literature review, and the gap: supervised FER needs many labels per
      emotion and degrades on rare classes
- [ ] Contributions, each with a reason it matters — not just a list
- [ ] High-level summary of the approach, details deferred

## 3. Method
- [ ] Formal problem statement: N-way K-shot episodes, support/query split
- [ ] **Pipeline/architecture diagram — marked VERY IMPORTANT in the brief**
- [ ] MAML formulation: inner update, outer meta-objective, first- vs
      second-order
- [ ] ConvNet4 backbone; note on BatchNorm without running statistics
- [ ] Dataset and preprocessing: FER2013, normalisation, augmentation
- [ ] Implementation details: optimiser, learning rates, episode counts,
      hardware, wall-clock time
- [ ] Justification of each design choice

## 4. Results
- [ ] **Baseline comparison** (required): SimpleCNN and supervised ConvNet4
      against MAML
- [ ] Main table: accuracy, balanced accuracy, macro-F1; few-shot numbers with
      95% CIs
- [ ] Held-out-emotion protocol — the central generalisation claim
- [ ] Confusion matrices, training curves
- [ ] Ablations (optional but "very positively evaluated"):
      K ∈ {1, 5}, inner steps ∈ {1, 5}, first- vs second-order,
      with/without class weighting
- [ ] Discussion of failure cases

## 5. Conclusion
- [ ] Summary of findings
- [ ] Contributions restated, not copy-pasted from the introduction
- [ ] Impact
- [ ] Limitations (FER2013 label noise, 48×48 resolution, single dataset)
- [ ] Future work

## 6. Contributions
- [ ] Per-student breakdown: coding, writing, ideas, literature review,
      methodology. Undisclosed under-contribution can affect the whole team's
      grade.

---

## Administrative
- [ ] Team of three registered on the SharePoint form
- [ ] Methodology + dataset confirmed with the teacher (~10-line summary)
- [ ] Repository public, URL live in the abstract
- [ ] Page count verified against the limit

## Numbers to fill in
Run `python scripts/make_report.py` after the experiments; it emits
`results/REPORT.md` with tables ready to transcribe.
