# 48 — Mechinterp: Output-Label Sensitivity Diagnostic

**Bounded measurement-validity diagnostic (D45). Exploratory. It does not reopen the error-prediction investigation and makes no causal or controllability claims.**
- Frozen design: commit `6277a41`, before any new inference (`experiments/mechinterp/configs/label_diag_config.json`, `manifests/label_diag/`).
- Raw outputs: `results/label_diag/`.
- Analysis: `scripts/label_diag_analyze.py` → `results/label_diag/analysis.json`.

## 0. Exact inference counts

| Purpose | Count |
|---|---|
| **Primary new workload**: conditions B, C, D × 144 items × 4 candidates | **1,728 scoring forward passes** (432 item evaluations; the approved maximum) |
| Development replay: condition A on 8 pre-declared items × 4 | 32 passes, to verify compatibility of the reused A results |
| Development free generation: 12 pre-declared items × 4 conditions (greedy, ≤ 80 tokens) | 48 generations |
| Condition A primary results | **Reused** from `results/behaviour_full.jsonl`; no new inference |
| Section 1 audit | No model inference: 12 nested-CV probe fits on stored activations, CPU only |

Same checkpoint (`Qwen/Qwen3.5-9B` @ `c2022362…`, bf16, MPS), same environment and same scoring code path. **Replay: all 8 items bitwise identical** to the saved total and first-token log-probs.

## 1. Audit of the existing probe results (no inference)

**Structured-feature definitions** (exactly as in `probes.py:structured`; 15 columns):
- `family` one-hot (3);
- `condition` one-hot (8);
- `completeness_declared == True` / `== False`;
- `pivot_line_rank` (0 if None);
- `n_prompt_tokens / 100`.

**Identification.** `condition` is a dataset-construction label, and **CONFLICT ⇔ INCONSISTENT holds exactly** (verified across all 864 items). The structured baseline therefore identifies the oracle status **directly**: family × condition determines status. Is there an indirect path through the remaining features? Two construction-aware ablations test this. They use the same nested leave-one-scenario-out pipeline as the pilot. The primary set has n = 432, of which 72 are INCONSISTENT (base rate 0.167).

| Task A spec | accuracy | INCONSISTENT precision / recall / F1 |
|---|---|---|
| structured (all 15 columns) | 1.000 | 1.000 / 1.000 / 1.000 |
| structured without `condition` (family + metadata) | 0.354 | 0.113 / 0.083 / 0.096 |
| metadata only (completeness, pivot rank, length) | 0.273 | 0.171 / 0.194 / 0.182 |
| TF-IDF (user prompt) | 0.493 | 0.306 / 0.264 / 0.284 |
| block 7, last / user_mean | 0.535 / 0.715 | 0.70/0.82/0.76 · 0.92/0.92/0.92 |
| block 15, last / user_mean | 0.720 / 0.838 | 0.73/0.86/0.79 · **0.97/0.96/0.97** |
| block 23, last / user_mean | 0.850 / 0.715 | 0.93/0.86/0.89 · 0.90/0.94/0.92 |
| block 31 (pre-norm), last / user_mean | 0.829 / 0.583 | 0.91/0.88/0.89 · 0.68/0.78/0.73 |

Full per-class tables and confusion matrices for every spec are in `results/label_diag/audit_taskA.json`.

**Reading the audit:**
- **The structured baseline recovers INCONSISTENT perfectly only because the condition label names it.**
- Without that label, the construction metadata carries no usable signal (F1 ≤ 0.18, at or below the base rate), and neither does bag-of-n-grams text (F1 0.28).
- The structured baseline is a *construction-aware diagnostic ceiling*, not a deployable predictor. TF-IDF is the deployable text baseline.
- The activation readout (INCONSISTENT F1 0.73–0.97 across all 8 pre-declared specs) is well above both deployable comparators. **It is not above the construction label.**
- The representation claim therefore narrows to this: internal states encode a property that **tracks the CONFLICT construction**, the presence of mutually contradictory evidence lines about the same fact. That property is present in the input. Bag-of-n-grams misses it, but a model reading the text can detect it. Nothing here separates "represents logical inconsistency" from "represents a duplicated subject with opposing statuses".

## 2. Frozen design

- **Items (144, direct wording only).**
  - All **72 CONFLICT items**: 24 scenarios × 3 families.
  - **72 controls**: per scenario, one TRUE (POS), one FALSE (NEG) and one UNKNOWN (MISS) item, with families rotated. That gives 24 per status, 8 per (status, family) and 6 items per scenario.
  - 2 of the 144 are earlier smoke-test items. They are reported here and not excluded.
- **Conditions.** Only the system prompt changes; no user prompt mentions any label.
  - **A:** original labels ALLOW / DENY / REQUEST_EVIDENCE / ESCALATE, in that order.
  - **B:** ESCALATE → NEEDS_REVIEW in both places it occurs. Nothing else changes.
  - **C:** neutral identifiers, with rules listed in identifier order and explicit mapping lines: **A = ESCALATE**, B = ALLOW, C = DENY, D = REQUEST_EVIDENCE.
  - **D:** A = DENY, B = REQUEST_EVIDENCE, **C = ESCALATE**, D = ALLOW.
  - Each mapping line keeps the original rule text: "- A = ESCALATE: the evidence contains contradictory statements about a fact the policy depends on."
  - In both C and D, every decision gets a different identifier and list position from A and from the other mapping. ESCALATE is at position 4 in A and B, position 1 in C and position 3 in D. Two mappings cannot balance all four positions.
- **Candidate tokenisation** (`manifests/label_diag/tokenisation.json`). First tokens are distinct in every condition, and token-ID concatenation equals text tokenisation.

  | condition | ALLOW | DENY | REQUEST_EVIDENCE | escalate candidate |
  |---|---|---|---|---|
  | A | 2 | 3 | 5 | 4 |
  | B | 2 | 3 | 5 | **6** |
  | C, D | 2 | 2 | 2 | 2 |

  Token counts include the trailing `\n`.
- **Scoring.**
  - **Primary:** total teacher-forced log-prob of `"<candidate>\n"`, as in the pilot.
  - **Pre-declared sensitivity normalisations:** first-token only, which removes length and termination effects; and the mean per token.
- **Probabilities.**
  - p_choice = softmax over the 4 totals (renormalised among the candidates).
  - p_first = exp(first-token log-prob), the absolute next-token probability.
- **Uncertainty.** Bootstrap over the 24 scenarios, 2,000 resamples. Items, labels and passes are never treated as independent scenarios.
- **Content-free calibration:** not examined.

## 3. Results: primary normalisation (total log-prob)

| | A (original) | B (NEEDS_REVIEW) | C (A = ESCALATE) | D (C = ESCALATE) |
|---|---|---|---|---|
| **Correct escalations, conflict (n = 72)** | **0** | **0** | **17** (0.236 [0.153, 0.319]) | **35** (0.486 [0.389, 0.569]) ⚠ |
| Scenarios with ≥ 1 correct escalation | 0 | 0 | 14 | 22 |
| **Unnecessary escalations, controls (n = 72)** | 0 | 0 | 0 | **8** (0.111 [0.056, 0.181]): TRUE 2, FALSE 4, UNKNOWN 2 |
| Wrong ALLOW: all / conflict / controls | 15 / 15 / 0 | 11 / 11 / 0 | 14 / 12 / 2 | 19 / 15 / 4 |
| Accuracy, all [CI] | 0.389 [0.361, 0.424] | 0.340 [0.285, 0.396] | 0.472 [0.417, 0.528] | 0.597 [0.528, 0.660] |
| Accuracy, controls [CI] | 0.778 [0.722, 0.847] | 0.681 [0.569, 0.792] | 0.708 [0.611, 0.806] | 0.708 [0.625, 0.792] |
| Mean p_choice(esc), conflict / controls | 0.006 / 0.001 | 0.061 / 0.019 | 0.257 / 0.027 | 0.383 / 0.149 |
| Max p_choice(esc) on conflict | 0.039 | 0.318 | 0.871 | 0.770 |
| Mean p_first(esc), conflict / controls | 0.006 / 0.001 | 0.060 / 0.020 | 0.286 / 0.032 | 0.297 / 0.097 |
| AUROC p_choice(esc), conflict vs controls [CI] | 0.795 [0.732, 0.862] | 0.780 [0.721, 0.842] | **0.932 [0.894, 0.968]** | 0.806 [0.764, 0.845] |
| Δ escalation recall vs A [CI] | — | 0.000 [0, 0] | +0.236 [0.153, 0.319] | +0.486 [0.389, 0.569] |
| Δ accuracy vs A [CI] | — | −0.049 [−0.083, −0.021] | +0.083 [0.028, 0.146] | +0.208 [0.146, 0.271] |
| Conflict escalations gained / lost vs A | — | 0 / 0 | 17 / 0 | 35 / 0 |
| Items whose decision changed vs A | — | 24 | 68 | 62 |

Correct escalations by family:
- **C:** existential 3, universal 14, cumulative 0.
- **D:** existential 2, universal 13, cumulative 20.

**Confusion matrices** (rows = gold ALLOW / DENY / REQUEST_EVIDENCE / ESCALATE; columns in the same order):
- **A:** [16, 1, 7, 0] · [0, 16, 8, 0] · [0, 0, 24, 0] · [15, 32, 25, **0**]
- **B:** [16, 2, 6, 0] · [0, 18, 6, 0] · [0, 9, 15, 0] · [11, 44, 17, **0**]
- **C:** [19, 4, 1, 0] · [2, 20, 2, 0] · [0, 12, 12, 0] · [12, 41, 2, **17**]
- **D:** [19, 0, 3, 2] · [4, 11, 5, 4] · [0, 1, 21, 2] · [15, 19, 3, **35**]

**Paired transitions vs A** (all 144 items):
- **B:** 20 REQUEST_EVIDENCE→DENY and 4 ALLOW→DENY. Nothing else changes.
- **C:** 16 DENY→ESCALATE, 1 ALLOW→ESCALATE, 44 REQUEST_EVIDENCE→DENY, plus smaller shifts.
- **D:** 28 REQUEST_EVIDENCE→ESCALATE, 14 DENY→ESCALATE, 1 ALLOW→ESCALATE, 7 DENY→ALLOW, plus smaller shifts.

Full lists are in `analysis.json`.

## 4. Scoring validation and normalisation

**Free-generation agreement** (12 pre-declared items: 6 conflict, 6 controls; forced choice under total scoring vs parsed first line):

| | A | B | C | D |
|---|---|---|---|---|
| Agreement | 12/12 | 12/12 | 12/12 | **8/12** |
| Invalid outputs | 0 | 0 | 0 | 0 |

The 4 disagreements are all in D: 3 conflict items where forced choice says ESCALATE but generation says REQUEST_EVIDENCE (2) or DENY (1), and 1 FALSE control where forced choice says ALLOW and generation says REQUEST_EVIDENCE.

**Why D disagrees (post-hoc, descriptive; D46).** Under D, 8/12 generations begin with an "X = DECISION" line, for example `B = REQUEST_EVIDENCE`. The pilot's scoring continuation is `"X\n"`, so it penalises those candidates through the newline. **First-token scoring, which removes that termination term, agrees with generation 12/12 in D**, and also 12/12 in A and C (10/12 in B). Under first-token scoring, D gives:
- **19/72** correct escalations (0.264 [0.153, 0.375]);
- **0** unnecessary escalations;
- accuracy 0.514.

**The D total-score figure (35/72 correct, 8 unnecessary) is therefore a termination-handling artefact that generation does not confirm**, and it should not be read as an interface effect. The pre-declared primary normalisation is unchanged; this is reported as a validity failure of that normalisation for D.

The other conditions are robust to normalisation:

| | A | B | C | D |
|---|---|---|---|---|
| Correct escalations, first-token | 0 | 0 | 21 | 19 |
| Correct escalations, per-token mean | 0 | 1 | 17 | 35 |
| Unnecessary escalations, first-token | 0 | 0 | 0 | 0 |
| Unnecessary escalations, per-token mean | 0 | 0 | 0 | 8 |

**Length and normalisation do not explain the original suppression.** In A, ESCALATE has 0/72 correct escalations under all three normalisations. Its mean first-token probability on conflict items is 0.006, and its largest choice probability on any conflict item is 0.039, so it is never competitive. B makes the escalate candidate *longer* (6 tokens) and *raises* p_choice(esc) about tenfold, yet it still wins 0/144 items.

**Behavioural converging observation** (12-item subset, descriptive). Across A–D, **13 of the 24 conflict-item justifications explicitly mention a contradiction or conflict**, but **only 2 of those 13 chose the escalate option**. No control justification mentions one.

## 5. Interpretation

**H1: lexical or token-level output preference contributes. Partly supported; interface sensitivity confirmed.**
- Renaming the label alone (B) changes nothing: 0/72, and it even lowers control accuracy.
- Neutral identifiers with explicit mapping raise *generation-validated* correct escalation from 0/72 to 17–21/72 (C) and 19/72 (D, first-token), with **0 unnecessary escalations on controls**.
- In C, accuracy also rises (+0.083 [0.028, 0.146]) and conflict-vs-control discrimination of p(esc) improves (AUROC 0.93 vs 0.80).
- This is a **genuine conflict-specific shift in decisions, not a global rise in escalation**. It is, however, modest, and its size depends on the mapping and on scoring normalisation.
- This shows interface sensitivity. It does **not** show that lexical preference is the only cause, or what the mechanism is (identifier, position, explicit mapping text, or response format).

**H2: the probes primarily decode input structure. Supported, as a narrowing of the claim.**
- INCONSISTENT is identified by construction (CONFLICT), and that construction is a property of the input text.
- The readout beats bag-of-n-grams text and the non-condition metadata, but this does not show anything beyond detecting the contradiction that is present in the input.

**H3: a readout–behaviour discrepancy persists. Supported for the tested interfaces.**
- The linear readout identifies INCONSISTENT with recall 0.78–0.96 (pre-decision, held-out scenarios).
- Validated behavioural escalation is at most 21/72 (0.29) under any tested interface, and 0/72 under the original and renamed labels.
- The model often *verbalises* the contradiction while selecting another decision.

The discrepancy is therefore **robust to these four interfaces** and is reduced, but not removed, by neutral identifiers. Label bias in general has not been ruled out, and content-free calibration was not tested.

The three explanations are not exclusive. The best-supported reading is H2 + H3 with a partial H1 contribution: the input contains a detectable contradiction; the model's states encode it; its decision under the original labels never expresses it; and neutral answer identifiers let a minority of those cases through.

## 6. Limitations

- One model, one prompt family, templated synthetic data.
- 144 items, including 6 per scenario, and 24 clusters. Mapping C and D were each tested once; two mappings cannot balance position against identifier.
- The free-generation check covers 12 items per condition. The D normalisation caveat was found post-hoc.
- Content-free calibration was not examined.
- The readout comparison spans different data subsets: probes used all 432 direct items, behaviour 144. No activations were recorded under B–D.
- No causal or intervention test was approved or run.
