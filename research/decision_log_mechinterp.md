# Decision Log: Mechanistic-Interpretability Pilot (excerpt D40–D47)

Verbatim excerpt of entries D40–D47 from the append-only decision log in the private research archive (commit `e6699e6`, `research/decision_log.md`). Entries D1–D39 concern earlier, separate investigations and are not included. Entries are written in the second person ("you approved") because they were recorded during the work for the research lead who set objectives and approvals.

## D40 — 2026-11-26 — Mechanistic-interpretability pilot: environment prepared; STOP for weight-download approval
This is a separate exploratory investigation (`experiments/mechinterp/`, reports/43–47); the earlier lines stay closed.
- Methods requirements are extracted from 2610.06452 (mentor paper; a task-specific TopK-SAE with 16,384 latents, K = 128, on pooled layer-24 states, confirmed), SAKIKO, Verdict Without the Rule, Auditing the Judge and TACIT.
- Qwen/Qwen3.5-9B: revision c2022362, Apache-2.0, ungated, 19.33 GB bf16, hybrid 32-layer stack (full attention every 4th layer, gated-delta-rule linear attention otherwise) plus a vision tower. The pure-PyTorch fallback for linear attention on MPS was verified in source.
- Locked environment: torch 2.15.0 (MPS OK), transformers 5.22.2 (qwen3_5 classes import).
- Pre-declared: layers hidden_states[8, 16, 24, 32]; primary last-generation-prompt-token representation; secondary user-token mean; forced-choice full-label scoring with a generation-agreement check.
- No weights downloaded. Awaiting user approval.

## D41 — 2026-11-26 — Mechinterp full run approved; amendment AM1 (made before any full-run result)
You approved the bounded 864-item run. The frozen config (`experiments/mechinterp/configs/pilot_config.json`) is executed unchanged.
**Amendment AM1 (additive only).** Task A (oracle-status decoding) also gets structured-feature and TF-IDF baselines. The reason: oracle status is a deterministic function of (family, condition), which is visible in the text, so activation decoding of status is only a sanity check unless it is compared with these baselines.
**Additional sensitivity analysis** (your request). Exclude the 12 previously inspected smoke-test items (DA01; 6 are direct-wording).
Nothing else changes: hypotheses, layers, primary set, baselines and stopping rules all stay as frozen.

## D42 — 2026-11-26 — Computational amendment AM2 (estimator unchanged)
The first probe run was aborted after 1,566 s: one activation spec plus 20 Task A permutations took about 75 s per out-of-fold pass, which projects to about 40 h for the frozen analysis. The only result seen was primary Task A act_7_last, macro-F1 0.551; the log is kept as `results/logs/probes_aborted_run1.log`.
**Fix.** Each standardised activation block (4,096 columns, fewer than 330 training rows) is now re-expressed in the orthonormal basis of its training-row space, computed by SVD of the training rows only, inside each fold. For L2 logistic regression this is an exact reparametrisation, so the estimator is unchanged.
**Checks.** A unit test shows equal held-out probabilities (difference < 1e-4). On real data, act_7_last gives macro-F1 0.555 vs 0.551, within solver tolerance and C-selection ties. Runtime is 12.9 s per pass.
Nothing else changes: layers, representations, baselines, C grid, CV, permutation counts and stopping rules are all as frozen. The full analysis is rerun from scratch.

## D43 — 2026-11-27 — Post-hoc analyses (declared after the Task B point estimates were seen)
These are labelled POST-HOC EXPLORATORY and do not change any pre-declared result.
1. **OOF rerun.** Rerun the pre-declared pipeline with `--perms 0 --save-oof` to obtain out-of-fold scores. This also checks determinism: the metrics must equal the main run's.
2. **Within-cell ranking.** Within (family, condition) cells that have at least 5 errors and at least 5 non-errors, report the AUROC of each score. Structured features are constant inside a cell, so this is the only available view of error-ranking signal beyond the item's structure. It is descriptive only.
3. **Smoke-item exclusion.** Rerun with `--exclude-smoke --perms 0` (your request).

## D44 — 2026-11-27 — Mechinterp pilot closed: technically feasible; no credible added value beyond item structure
**Primary set** (432 direct items): 203 errors, and all 24 scenarios contain errors. The model never outputs ESCALATE.
- **Task B (error prediction).** The best activation probe reaches AUROC 0.919 (block 31 last) and 0.918 (block 23 last). Text + margin reaches 0.838 (pre-declared incremental ΔAUROC +0.08, CI excludes 0). Structured features reach 0.955, and structured + margin 0.969.
- **Within cells** (post-hoc), activations show no error ranking. The pre-declared stop condition "no credible added value" is met with respect to item structure.
- **Task A (status decoding).** Activations reach macro-F1 up to 0.852, against TF-IDF 0.454; shuffled controls give about 0.25.
- **Post-hoc lead.** The INCONSISTENT status is decodable (recall 0.86–0.96) although the model never outputs ESCALATE. This is not a causal claim.
**Sensitivity.** Excluding the smoke items changes AUROC by at most 0.003.
**Status.** No pivot. Further work needs your approval (reports/47 §5).

## D45 — 2026-11-27 — Final measurement diagnostic: output-label sensitivity (frozen before inference)
You approved one bounded measurement-validity diagnostic. It is not a new investigation, and the pilot's negative error-prediction conclusion stands.
**Question.** Is the INCONSISTENT-readout vs ESCALATE-behaviour discrepancy explained by output-label preference (H1) or dataset structure (H2), or does it persist (H3)?
**Frozen design** (`experiments/mechinterp/configs/label_diag_config.json`, `manifests/label_diag/`):
- 144 direct items: 72 CONFLICT, plus 72 POS/NEG/MISS controls, 24 per status, families rotated, 6 per scenario.
- Conditions:
  - A: original interface; saved results reused after an 8-item bitwise replay.
  - B: ESCALATE → NEEDS_REVIEW.
  - C and D: neutral identifiers A–D under two mappings, with rules listed in identifier order and explicit mapping lines.
- Scoring: primary is the total teacher-forced log-prob, as in the pilot. Sensitivity normalisations: first-token and per-token mean.
- Workload:
  - primary: 1,728 scoring passes (the approved maximum);
  - development: 32 replay passes plus 48 greedy generations on 12 pre-declared items × 4 conditions.
- Metrics and a scenario-cluster bootstrap are pre-declared. Content-free calibration is not examined.
The section 1 audit (existing probe results, structured-feature definitions) needs no inference.

## D46 — 2026-11-27 — Label diagnostic results; post-hoc normalisation check; final close
**Inference.** 1,728 primary scoring passes (B, C, D), 32 replay passes (A; bitwise equal to the saved results) and 48 free generations. A was reused. No other inference.
**Audit.** The structured baseline identifies INCONSISTENT only through the condition one-hot (CONFLICT ⇔ INCONSISTENT). Without it, INCONSISTENT F1 is ≤ 0.18; TF-IDF gives 0.28 and activations 0.73–0.97.
**Results** (primary total-score normalisation):

| Condition | Correct escalations (of 72) | Unnecessary escalations (of 72 controls) |
|---|---|---|
| A | 0 | 0 |
| B (NEEDS_REVIEW) | 0 | 0 |
| C (A = ESCALATE) | 17 | 0 (AUROC 0.93) |
| D (C = ESCALATE) | 35 | 8 |

**Post-hoc (descriptive).** Free generation disagrees with D total-score forced choice on 4/12 items, because the model writes "X = DECISION" first lines. First-token scoring agrees 12/12 and gives D 19/72 correct and 0 unnecessary. The D total-score figure is reported as a termination artefact; the primary normalisation is unchanged.
**Interpretation.**
- H1 (lexical/output preference contributes): partial — interface sensitivity, not sole cause.
- H2 (probes decode input structure): supported — the representation claim narrows to tracking the CONFLICT input construction.
- H3 (readout–behaviour discrepancy persists): supported across the 4 tested interfaces.
**Status.** The error-prediction negative stands. This is the final pre-application experiment, and the pilot is closed (reports/48–49).

## D47 — 2026-11-27 — Final integrity and reproducibility check (housekeeping; no inference, no protocol change)
- **Raw condition-A scores preserved.** `experiments/mechinterp/results/behaviour_full.jsonl` (864 items) is tracked in Git. It was committed in `56a37be` and has not changed since (SHA-256 `329e3623…0540`). The SHA-256s of the raw results, diagnostic items, source dataset and analysis are now recorded in `experiments/mechinterp/manifests/raw_results.sha256`.
- **Independent recomputation.** The new `scripts/verify_integrity.py` (standard library only; also run by `tests/test_integrity.py`) recomputes the following from the raw log-probs, and all agree with reports 44/47/48 and `analysis.json`:
  - all 144 diagnostic items match the raw file by `item_id`, and their user prompts match the source by SHA-256;
  - stored decision = argmax for 864/864 items;
  - CONFLICT items: 0/72 correct escalations and 15/72 wrong ALLOW (DENY 32, REQUEST_EVIDENCE 25);
  - controls: 0/72 unnecessary escalations;
  - overall: 0/864 ESCALATE, max p_choice(ESCALATE) 0.043;
  - the 8-item replay is bitwise equal to the raw file.
- **Not committed:** the activations (54 MB; gitignored; 9/9 shard SHA-256s re-verified against `extraction_validation.json`) and the model weights (in the HF cache).
- **Documentation corrections:**
  1. Date error: the README said all phases took place on 2026-11-25/26. D43–D46 were committed on 2026-11-27 IST. Corrected, and the status date updated.
  2. Added an explicit statement (README, `experiments/mechinterp/README.md`, a `reports/49` §3 addendum) that the results are text-level authorization judgments, not tool executions.
  3. The mechinterp README now covers reports 48–49, the diagnostic reproduction commands, artifact locations and AI assistance.
- No experiment was run, and no result, protocol or historical commit was changed.
