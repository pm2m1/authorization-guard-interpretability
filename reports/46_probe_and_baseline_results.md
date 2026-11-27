# 46 — Mechinterp Pilot: Probe and Baseline Results

**EXPLORATORY feasibility pilot. Nothing here is confirmatory. Decodability does not mean the model uses that information causally.**

Sources:
- `experiments/mechinterp/results/probe_results.json`: the pre-declared run, 17,053 s.
- `probe_results_oof_rerun.json`: a deterministic rerun with out-of-fold dumps.
- `probe_results_excl_smoke.json`: the sensitivity analysis.
- `posthoc_*.json`: post-hoc analyses (D43).

Every pre-declared layer and representation is reported. None was selected after the results were seen.

**Pipeline** (frozen config, plus AM1, which added baselines, and AM2, an exact computational reparametrisation):
- L2 logistic regression with balanced class weights;
- C chosen from {1e-4, 1e-3, 1e-2, 1e-1, 1} by an inner 4-fold group split over the training scenarios;
- outer leave-one-scenario-out validation over the 24 bases;
- standardisation, TF-IDF and the SVD basis all fitted on training folds only. Outer folds were asserted to be scenario-disjoint.

**Intervals and controls:**
- 95% intervals come from a 2,000-sample bootstrap over the 24 scenario clusters.
- Shuffled-label controls use the same pipeline with permuted labels: 100 permutations for Task B, 20 for Task A.
- A rerun with identical inputs reproduced every metric exactly (max |Δ| = 0.000).

## 1. Task A: decoding oracle status (primary, direct wording, n = 432; 4 classes)

| Representation | accuracy [95% CI] | macro-F1 | shuffled macro-F1, mean / max |
|---|---|---|---|
| structured (family, condition, …) | 1.000 | 1.000 | — |
| TF-IDF (user prompt) | 0.493 [0.479, 0.507] | 0.454 | — |
| block 7, last | 0.535 [0.498, 0.569] | 0.555 | 0.246 / 0.286 |
| block 7, user_mean | 0.715 [0.664, 0.766] | 0.737 | 0.242 / 0.283 |
| block 15, last | 0.720 [0.681, 0.757] | 0.717 | 0.243 / 0.287 |
| block 15, user_mean | 0.838 [0.792, 0.877] | **0.852** | 0.250 / 0.316 |
| block 23, last | 0.850 [0.824, 0.875] | 0.845 | 0.241 / 0.302 |
| block 23, user_mean | 0.715 [0.678, 0.750] | 0.733 | 0.246 / 0.298 |
| block 31 (pre-norm), last | 0.829 [0.803, 0.854] | 0.828 | 0.246 / 0.304 |
| block 31 (pre-norm), user_mean | 0.583 [0.544, 0.618] | 0.597 | 0.246 / 0.285 |

**How to read this.** Oracle status is a deterministic function of the item's design (family × condition), which is why the structured baseline is perfect. Decoding status from activations is therefore a **sanity check, not a discovery**. What it does show is that the middle and late layers recover the item's logical class far better than a bag-of-n-grams model trained on other scenarios: 0.85 vs 0.45 macro-F1. Every activation probe beats its shuffled control by a wide margin. The dual-wording stress set gives the same pattern, with macro-F1 0.548–0.848 (`probe_results.json`).

**Post-hoc (D43): per-class recall for the primary set.** The INCONSISTENT class consists of CONFLICT items, whose gold answer is ESCALATE.

| | TRUE | FALSE | UNKNOWN | **INCONSISTENT** |
|---|---|---|---|---|
| TF-IDF | 0.40 | 0.64 | 0.53 | **0.26** |
| block 23, last | 0.83 | 0.87 | 0.83 | **0.86** |
| block 15, user_mean | 0.75 | 0.82 | 0.92 | **0.96** |
| range across all 8 activation specs | 0.41–0.83 | 0.49–0.87 | 0.57–0.92 | **0.78–0.96** |

## 2. Task B: predicting errors (primary, n = 432, 203 errors, all 24 scenarios contain errors)

The label is 1 when the forced-choice decision differs from the oracle decision. The base rate is 0.470. Recall@k is the share of all errors found when reviewing the top-k% of items by score. The risk–coverage row gives the error rate among the 25%, 50% and 75% of items with the lowest risk scores.

| Score | AUROC [95% CI] | AP [95% CI] | R@5% / 10% / 20% | risk at 25 / 50 / 75% coverage |
|---|---|---|---|---|
| margin (−margin) | 0.689 [0.658, 0.720] | 0.597 | 0.06 / 0.12 / 0.25 | 0.18 / 0.34 / 0.44 |
| confidence (1 − max p) | 0.703 [0.676, 0.727] | 0.627 | 0.07 / 0.14 / 0.26 | 0.18 / 0.32 / 0.43 |
| TF-IDF | 0.732 [0.704, 0.759] | 0.737 | 0.09 / 0.18 / 0.35 | 0.31 / 0.31 / 0.35 |
| **text + margin** (pre-declared reference) | **0.838 [0.822, 0.854]** | 0.806 | 0.10 / 0.19 / 0.37 | 0.07 / 0.21 / 0.34 |
| **structured** | **0.955 [0.938, 0.973]** | **0.968** | 0.11 / 0.21 / 0.42 | 0.08 / 0.05 / 0.29 |
| **structured + margin** | **0.969 [0.957, 0.981]** | **0.972** | 0.11 / 0.21 / 0.42 | 0.04 / 0.06 / 0.29 |
| block 7, last | 0.844 [0.810, 0.876] | 0.839 | 0.10 / 0.20 / 0.40 | 0.14 / 0.19 / 0.33 |
| block 7, user_mean | 0.845 [0.814, 0.879] | 0.862 | 0.11 / 0.21 / 0.40 | 0.18 / 0.20 / 0.31 |
| block 15, last | 0.860 [0.830, 0.892] | 0.876 | 0.11 / 0.21 / 0.40 | 0.15 / 0.19 / 0.32 |
| block 15, user_mean | 0.902 [0.879, 0.926] | 0.915 | 0.11 / 0.21 / 0.42 | 0.11 / 0.14 / 0.30 |
| block 23, last | 0.918 [0.895, 0.940] | 0.916 | 0.11 / 0.21 / 0.41 | 0.05 / 0.13 / 0.31 |
| block 23, user_mean | 0.869 [0.837, 0.903] | 0.894 | 0.11 / 0.21 / 0.42 | 0.12 / 0.19 / 0.30 |
| block 31, last | **0.919 [0.905, 0.935]** | 0.918 | 0.11 / 0.21 / 0.41 | 0.06 / 0.12 / 0.31 |
| block 31, user_mean | 0.825 [0.790, 0.863] | 0.841 | 0.11 / 0.21 / 0.40 | 0.16 / 0.22 / 0.34 |

Activation + text + margin combinations give 0.839–0.919, essentially the same as activations alone (`probe_results.json`).

Notes on the table:
- The recall-budget ceiling is low because errors make up 47% of items. Reviewing 20% of items can find at most 0.43 of the errors, and the best scores (0.42) reach that ceiling.
- **Shuffled-label controls:** mean AUROC 0.498–0.508, 95th percentile at most 0.562, maximum 0.632 over 100 permutations per spec. Every real activation AUROC (≥ 0.825) is far above every permutation maximum.

**Pre-declared incremental test vs text + margin** (ΔAUROC with cluster-bootstrap 95% CI):

| block | last | user_mean |
|---|---|---|
| 7 | +0.006 [−0.019, 0.031] | +0.008 [−0.020, 0.036] |
| 15 | +0.022 [−0.011, 0.056] | **+0.064 [0.042, 0.087]** |
| 23 | **+0.080 [0.052, 0.104]** | **+0.031 [0.002, 0.061]** |
| 31 | **+0.081 [0.064, 0.099]** | −0.012 [−0.047, 0.024] |

ΔAP is positive for most specs, from +0.033 to +0.112.

**Against the structured baseline, no activation spec comes close.** The best, 0.919, is below structured (0.955) and structured + margin (0.969), and outside their intervals. Activations + structured was not a pre-declared combination and was not run.

**Pre-declared within-status AUROC** (pooled out-of-fold scores):
- TRUE stratum: text + margin 0.987; activations 0.924–0.984.
- FALSE stratum: text + margin 0.882; activations 0.754–0.905. The best is block 31 last at 0.905, and block 23 last at 0.903.
- UNKNOWN: 4 errors, so not estimable.
- INCONSISTENT: every item is an error, so not estimable.

**Post-hoc within-cell analysis** (D43). Structured features are constant within a cell; only cells with at least 5 errors and 5 non-errors qualify.

| cell (primary) | errors / non-errors | margin | text + margin | activations, range over 8 specs |
|---|---|---|---|---|
| existential MISS_LCWA | 9 / 15 | 0.68 | 0.51 | 0.40–0.68 |
| universal MISS_LCWA | 18 / 6 | **0.13** | 0.14 | 0.30–0.57 |

Inside these cells the activations show **no error-ranking signal**. In universal MISS_LCWA, margin ranks errors *below* correct answers (AUROC 0.13): the wrong answers there, mostly wrong ALLOW, are **more confident** than the correct ones. Cells this small (n = 24) support description only.

## 3. Stress set (dual wording, n = 432, 262 errors, all 24 scenarios contain errors)

| | AUROC |
|---|---|
| margin | 0.676 |
| text + margin | 0.741 |
| structured | 0.868 |
| structured + margin | 0.894 |
| activations | 0.707–0.895 (best: block 23 last 0.895, block 31 last 0.883) |

- ΔAUROC vs text + margin ranges up to +0.154 [0.125, 0.185] (block 23 last).
- Shuffled-label controls: mean 0.489–0.507, maximum 0.599.
- Within cells (post-hoc): existential MISS (15 / 9) gives activations 0.21–0.59 and TF-IDF 0.71; universal MISS_IRREL (6 / 18) gives margin 0.98 and block 23 last 0.95.

The ordering matches the primary set: structured ≥ best activations > text + margin > margin.

## 4. Sensitivity: excluding the 12 inspected smoke-test items

Primary set: n = 426, 203 errors, 24 scenarios with errors. Every AUROC changes by at most 0.003; for example, block 31 last stays at 0.919, structured goes 0.955 → 0.954 and text + margin 0.838 → 0.836. Task A macro-F1 changes by at most 0.022 (TF-IDF). The stress set changes by at most 0.010. **No conclusion depends on the smoke items.**

## 5. Leakage checks

- Outer folds are asserted scenario-disjoint.
- All fitted preprocessing happens inside folds: scaler, TF-IDF vocabulary and IDF, and the SVD basis.
- C is selected only on training scenarios.
- Shuffled-label controls are at chance.
- The structured features include `n_prompt_tokens` and `pivot_rank`, which are design metadata. No oracle label or decision enters any feature.
- Margin and confidence come from the same forward pass as the decision. That is legitimate for an error detector, but it is not independent of the output.

## 6. Runtime and memory

| Stage | Time | Memory / hardware |
|---|---|---|
| Behaviour + capture | 4,227 s wall (about 2,300 s of compute; one idle-sleep stall) | MPS peak 25.6 GB |
| Probes (pre-declared, with 1,920 permutation passes) | 17,053 s | 15 CPU cores |
| Each rerun without permutations | about 5–10 min | 15 CPU cores |
| Aborted first probe run | 1,566 s, before AM2 | — |
