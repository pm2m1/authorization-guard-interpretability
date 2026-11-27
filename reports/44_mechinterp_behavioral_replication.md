# 44 — Mechinterp Pilot: Behavioural Labels from the bf16 Checkpoint

**EXPLORATORY.** This uses the existing 864 exploratory items. It is not confirmatory and no new scenarios were generated. The labels were regenerated from scratch with `Qwen/Qwen3.5-9B` at revision `c2022362…`, in bf16 on MPS: the same weights, precision and procedure later used for activation extraction. No Ollama Q4_K_M labels are reused. Config: `experiments/mechinterp/configs/pilot_config.json` (sha256 `4527b7b2…`), frozen before the run. Raw results: `experiments/mechinterp/results/behaviour_full.jsonl` (864 rows plus a metadata row) and `results/dev_agreement.jsonl`.

## 1. Interface

**Forced choice.** For each label in ALLOW, DENY, REQUEST_EVIDENCE and ESCALATE, the model is fed the prompt followed by the label and a newline, and the total teacher-forced log-probability of that continuation is recorded.
- The prompt is the official chat template with `enable_thinking=False`.
- Labels are multi-token: 2, 3, 5 and 4 tokens respectively.
- Decision = argmax. Margin = top minus second. Confidence = maximum of the softmax over the four totals.

Forced choice cannot produce an invalid output. Invalid outputs can only occur in free generation, and they are counted separately.

**Pre-declared generation agreement check** (48 items; `random.Random(20261126)`; greedy decoding, at most 80 new tokens, pilot parser). It ran **before** the full run.

| n | forced = generated | generation invalid / ambiguous / empty | parse status | overlap with smoke items |
|---|---|---|---|---|
| 48 (22 direct, 26 dual) | **48/48 (100%)** | 0 | 48 STRICT | 0 |

The 90% stopping rule is passed. Together with the 12/12 in the smoke test, the forced-choice interface reproduces free-generation decisions on 60/60 inspected items. The first-token-only decision agrees with the full-label decision on 423/432 direct and 417/432 dual items. The full-label rule is the pre-declared one.

**Reproducibility.** The 48 development items were scored twice: once in the agreement-check run and once in the full run, each in a separate process. All 48 label log-probabilities are **bitwise identical**.

## 2. Primary set: direct wording (432 items, 24 scenarios)

**Accuracy 229/432 = 0.530. Errors: 203. Every one of the 24 scenarios contains errors: 7–10 per scenario, median 9.** Wrong ALLOW (fail-open): 24.

Confusion matrix (rows = oracle decision, columns = model decision):

| gold \ model | ALLOW | DENY | REQUEST_EVIDENCE | ESCALATE | n |
|---|---|---|---|---|---|
| ALLOW | **71** | 1 | 48 | 0 | 120 |
| DENY | 9 | **90** | 69 | 0 | 168 |
| REQUEST_EVIDENCE | 0 | 4 | **68** | 0 | 72 |
| ESCALATE | 15 | 32 | 25 | **0** | 72 |

**The model never chose ESCALATE.** Across all 864 items, the largest softmax probability it gave ESCALATE was 0.043. As a result, all 72 CONFLICT items (gold ESCALATE) are errors, and 15 of them resolve to ALLOW.

Errors by family × condition (24 items per cell):

| family | POS | NEG | MISS | MISS_LCWA | MISS_IRREL | CONFLICT | MISS_CAP | MISS_FLOOR |
|---|---|---|---|---|---|---|---|---|
| existential | 1 | 0 | 3 | 9 | 0 | 24 | — | — |
| universal | 1 | 1 | 1 | 18 | 3 | 24 | — | — |
| cumulative | 23 | 23 | 0 | — | — | 24 | 24 | 24 |

**Structure of the errors.** Errors are almost entirely determined by the cell:
- 5 of the 18 cells are entirely wrong, and 3 are entirely right.
- The remaining 10 "mixed" cells contain 240 items and 83 errors. Most of those errors are concentrated in cumulative POS and NEG (23/24 each) and in universal MISS_LCWA (18/24).
- Only **existential MISS_LCWA (9/24)** and **universal MISS_LCWA (18/24)** show substantial variation within a cell.

So most of the error signal is predictable from structured input features alone. Report 46 tests whether activations add anything beyond that.

**Cumulative family.** The model fails it almost totally: it answers REQUEST_EVIDENCE or DENY regardless of whether the threshold was met. The only exception is cumulative MISS, which is 0/24 errors because its gold answer is REQUEST_EVIDENCE. This is the main contributor to errors that are not CONFLICT.

**Wrong ALLOW (24).** 13 are existential CONFLICT, 2 universal CONFLICT, 7 universal MISS_LCWA (gold DENY), 1 universal NEG and 1 cumulative NEG.

**Sensitivity analysis excluding the smoke-test items.** The 12 inspected items include 6 direct-wording ones. Without them: n = 426, errors 203, all 24 scenarios still contain errors. None of the 6 excluded direct items was an error, so the counts of errors and of scenarios containing errors are unchanged.

## 3. Exploratory stress set: dual wording (432 items; reported separately and not primary)

The dual wordings include the scope-ambiguous existential dual and the polarity-ambiguous "policy satisfied" framing (`reports/14`). Their oracle labels are therefore contestable.

- **Accuracy 170/432 = 0.394. Errors 262. All 24 scenarios contain errors (10–13 each).** Wrong ALLOW: 24.
- Confusion (rows ALLOW / DENY / REQUEST_EVIDENCE / ESCALATE, columns the same): [24, 40, 56, 0], [18, 92, 58, 0], [3, 15, 54, 0], [3, 38, 31, 0].
- The extra errors concentrate in **existential POS (24/24), existential MISS_IRREL (24/24) and existential MISS (15/24)**: exactly the cells whose dual reading is scope-ambiguous. **Universal MISS_LCWA** reaches 22/24, of which 16 are wrong ALLOW.
- Excluding the smoke items: n = 426, 259 errors.

## 4. Comparison with earlier runs

The earlier exploratory runs used **Ollama Q4_K_M free generation**. That is a different quantisation, runtime and interface, so its labels are neither reused nor compared as ground truth. The bf16 labels here are the only labels used for probes.

## 5. Limitations

- The data are synthetic, generated from templates, and were previously inspected. They are exploratory material only.
- Labels come from one checkpoint and one deterministic forced-choice interface. Validity rests on the oracle, and the dual-wording oracle labels are contestable.
- The ESCALATE collapse means "error" is partly a stand-in for "gold = ESCALATE".
- The total wall time of 4,227 s includes one ~16-minute idle-sleep stall: the maximum single forward pass was 987 s, against a median of 0.66 s. The compute-only estimate is about 2,300 s.
