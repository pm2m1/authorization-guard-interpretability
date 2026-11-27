# Results Index

Each claim below is mapped to the preserved file and JSON key it comes from, and to the report section that discusses it. Paths are relative to `experiments/mechinterp/` unless they start with `reports/`. Intervals are 95% percentile intervals from a 2,000-resample bootstrap over the 24 base scenarios.

**Status labels:**
- **P**: pre-declared in a configuration frozen before the relevant results were seen.
- **PH**: post-hoc; declared after results were seen (decision-log D43, D46).
- **D**: descriptive.

## Behaviour: bf16 forced choice, original labels

| Claim | Value | Status | Source |
|---|---|---|---|
| Items / base scenarios | 864 / 24. The 432 direct-wording items are the primary set; the 432 dual-wording items are a stress set. | P | `../../data/synthetic/exploratory_v1/items.jsonl`; `configs/pilot_config.json` |
| Primary accuracy | 229/432 = 0.530; 203 errors; errors in 24/24 scenarios (7–10 each) | P | `results/probe_results.json` → `primary_direct.behaviour`; reports/44 §2 |
| ESCALATE chosen | 0/864; max softmax probability 0.043 | D | `results/behaviour_full.jsonl`; recomputed by `scripts/verify_integrity.py` |
| CONFLICT items (72 direct) | 0/72 escalated (also 0/72 under first-token scoring); ALLOW 15, DENY 32, REQUEST_EVIDENCE 25 | P | same; also `results/label_diag/analysis.json` → `norm_total.A` |
| Forced choice vs greedy generation | 60/60 agreement (12 smoke items + 48 development items) | P | `results/smoke_test.json`, `results/dev_agreement_summary.json`; reports/44 §1 |
| Bitwise reproducibility | 48 development items re-scored in a separate process give identical log-probs | D | reports/44 §1, reports/45 §2 |

## Task B: error prediction (primary set, n = 432, base rate 0.470)

Source: `results/probe_results.json` → `primary_direct.taskB_error_prediction`. Discussion: reports/46 §2 and reports/47 §2.

| Score | AUROC | Status |
|---|---|---|
| Output margin | 0.689 [0.658, 0.720] | P |
| TF-IDF | 0.732 [0.704, 0.759] | P |
| Text + margin (pre-declared reference) | 0.838 [0.822, 0.854] | P |
| Best activation probe (block 31 pre-norm, last token) | **0.919** [0.905, 0.935] | P |
| Activation probes, all 8 | 0.825–0.919 | P |
| Construction-aware structured features | 0.955 [0.938, 0.973] | P |
| Construction-aware structured + margin | **0.969** [0.957, 0.981] | P |
| Shuffled-label controls (100 permutations per probe) | mean 0.498–0.508, max 0.632 | P |
| ΔAUROC vs text + margin, block 31 last / block 23 last | +0.081 [0.064, 0.099] / +0.080 [0.052, 0.104] | P |
| Within-cell AUROC of activations: existential / universal MISS_LCWA | 0.40–0.68 / 0.30–0.57 (n = 24 each) | PH, D |

**Conclusion** (P stop rule, reports/47): the activations add no credible value beyond item structure.

## Task A: decoding oracle status (primary set, 4 classes)

| Claim | Value | Status | Source |
|---|---|---|---|
| Macro-F1: activations / TF-IDF / shuffled | 0.555–0.852 / 0.454 / about 0.25 | P | `results/probe_results.json` → `primary_direct.taskA_oracle_status` |
| INCONSISTENT (CONFLICT) recall, best | **0.958** (69/72), block 15, mean-pooled | PH | `results/posthoc_taskA_per_class.json` → `act_15_user_mean` |
| INCONSISTENT recall, earliest layer | **0.917** (66/72), block 7, mean-pooled | PH | same → `act_7_user_mean` |
| INCONSISTENT recall, all 8 probes / TF-IDF | 0.778–0.958 / 0.264 | PH | same |
| Construction-aware baseline without the condition label | INCONSISTENT F1 at most 0.182 | PH (audit) | `results/label_diag/audit_taskA.json`; reports/48 §1 |

## Answer-interface diagnostic (144 items: 72 CONFLICT, 72 controls)

Source: `results/label_diag/analysis.json`, keys `norm_total` (primary), `norm_first` and `norm_mean`. Discussion: reports/48.

| Condition | Correct escalations (of 72), total / first-token | Unnecessary escalations (of 72), total / first-token | Wrong ALLOW on controls, total / first-token | Control accuracy, total / first-token | Status |
|---|---|---|---|---|---|
| A: original labels | 0 / 0 | 0 / 0 | 0 / 0 | 0.778 / 0.750 | P |
| B: ESCALATE renamed NEEDS_REVIEW | 0 / 0 | 0 / 0 | 0 / 0 | 0.681 / 0.694 | P |
| C: neutral letters, A = ESCALATE | 17 / **21** | 0 / 0 | 2 / 2 | 0.708 / 0.736 | P |
| D: neutral letters, C = ESCALATE | 35 / 19 | 8 / 0 | 4 / 1 | 0.708 / 0.764 | P (values); PH (validity) |

- **Scoring artefact in D.** Total scoring gives 35/72 and first-token scoring 19/72, a **22.2-percentage-point** gap. Free generation agrees with total-score forced choice on only 8/12 development items in D, but with first-token scoring on 12/12, because the model answers "X = DECISION" rather than "X". The total-score D figure is treated as an artefact (PH, D46). The first-token figure is the better-supported sensitivity result, not an independently validated replacement procedure. Source: `results/label_diag/freegen.jsonl`, `freegen_by_norm.json`.
- **Discrimination in C.** AUROC of p(escalate), conflict vs controls: 0.932 [0.894, 0.968] (P).
- **Verbalised contradictions.** 13 of 24 conflict-item justifications mention a contradiction; 2 of those 13 escalate (D).

## Test and integrity status (this snapshot)

The snapshot has 69 tests: 60 at the root and 9 in `experiments/mechinterp`. The full research archive has 106; the difference covers code that is not in this snapshot. Checksum and integrity results are in [PROVENANCE.md](PROVENANCE.md#verification-of-this-snapshot).
