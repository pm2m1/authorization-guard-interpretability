# 47 — Mechinterp Pilot: Feasibility Decision

**Decision.**
- **Technically FEASIBLE.**
- **Error detection: NO CREDIBLE ADDED VALUE BEYOND ITEM STRUCTURE** (the pre-declared stop condition is met).
- **One post-hoc lead** (§3) would need fresh, independent data and causal tests.

This pilot is exploratory: one model, 864 previously inspected synthetic items, no confirmatory claims. Nothing was pushed or published. No additional models or SAEs were downloaded. There was no steering, no new scenarios and no preregistration.

## 1. Exact experimental counts

| Stage | Count |
|---|---|
| Model | `Qwen/Qwen3.5-9B` @ `c2022362…`, bf16, MPS. 4 shards, SHA-256 equal to the LFS OIDs. |
| Smoke test | 12 items. Each scored with 4 forward passes; 4 of them scored twice; 12 greedy generations; one extra hooks-on / hooks-off pass. |
| Generation agreement check | 48 items: 48 × 4 scoring passes plus 48 greedy generations |
| Full run | 864 items × 4 teacher-forced passes = **3,456 forward passes**. 864 × 4 blocks × 2 representations × 4,096 activations captured. |
| Primary analysis | 432 direct-wording items, 24 scenarios, **203 errors; all 24 scenarios contain errors (7–10 each)** |
| Stress analysis | 432 dual-wording items, 262 errors, 24/24 scenarios contain errors |
| Probe fits | 2 sets × (Task A: 10 specs + 160 permutation passes; Task B: 20 specs + 800 permutation passes), each a 24-fold outer / 4-fold inner nested CV. Plus 2 reruns without permutations and 1 post-hoc per-class run. |
| Amendments | AM1: added Task A baselines. AM2: exact reparametrisation for runtime. D43: post-hoc analyses, labelled as such. |

## 2. Pre-declared stop conditions

| Condition | Status |
|---|---|
| Runtime impractical | **Not met.** About 40 min of compute for the behaviour run and 4.7 h for the probes, after AM2. |
| Too few errors (< 30) | **Not met.** 203 primary errors, across 24 scenarios. |
| Interface changes the task (agreement < 90%) | **Not met.** Forced choice equals greedy generation on 48/48 items (60/60 including the smoke test). |
| No credible added value | **Met, with respect to item structure.** See below. |

**Added-value judgement.** The pre-declared incremental test passes. Activations at blocks 23 and 31 (last token) beat text + margin by ΔAUROC +0.08 (CI excludes 0), and so does block 15 user_mean (+0.064).

But this added value is not credible as an error signal:
1. **Structured item features beat every activation probe**: 0.955 alone and 0.969 with margin, against at most 0.919.
2. **The model's errors are almost entirely determined by the cell** (report 44): it never outputs ESCALATE, and it fails the cumulative family wholesale.
3. **Inside the only cells that vary**, the activations rank errors no better than chance (post-hoc).
4. Task A shows the same layers decode the item's logical class far better than TF-IDF. The simplest explanation is that the activation probes **recover the item's structure better than bag-of-n-grams does**, and errors follow from that structure.

In short, the activations act as a better text encoder here. They have not been shown to be an independent error signal.

## 3. Strongest supported observation (exploratory; post-hoc for the per-class part)

**The model never outputs ESCALATE**: P(ESCALATE) is at most 0.043 on every one of the 864 items, so all 72 primary CONFLICT items are wrong and 15 of them are fail-open ALLOW. **Yet the INCONSISTENT status of those same items is linearly decodable** at the pre-decision position. Recall is 0.86 at block 23 last and 0.96 at block 15 user_mean, against 0.26 for TF-IDF, under leave-one-scenario-out validation with a pre-declared pipeline. Shuffled-label macro-F1 is 0.24–0.25.

This is a **"represented but not acted on" pattern**. Neither decodability nor its gap from behaviour shows that the model *uses* the representation, or that intervening on it would elicit ESCALATE.

Cautions:
- Conflict may be signalled by surface features that a non-bag-of-words lexical model would catch.
- The per-class breakdown was examined after the main results were seen.
- The data are templated and were previously inspected.

**Secondary observation.** In universal MISS_LCWA, wrong answers (mostly wrong ALLOW) carry **larger** margins than correct ones (within-cell AUROC 0.13), so output confidence would systematically miss these errors. n = 24; descriptive only.

## 4. Limitations

- **Data:** synthetic, templated, 24 base scenarios, and already inspected across earlier phases. The structured features encode the oracle design exactly, and they would not be available in deployment.
- **Labels and interface:** one model, one prompt, one deterministic forced-choice interface. The dual-wording oracle labels are contestable.
- **Representations:** 4 layers (all full-attention blocks) and 2 token positions. Nothing on the 24 linear-attention layers or on heads. Linear probes only. Block 31 is pre-norm.
- **Error label:** dominated by the ESCALATE collapse and cumulative-family failure, so "error prediction" is largely "cell prediction".
- **Statistics:** 24-cluster bootstrap intervals are approximate; within-cell analyses have n = 24.
- **Process:** AM2 changed the computation after one intermediate result was seen. It is exact, and its effect is 0.004 F1 from solver tolerance.
- **Disclosure:** all design and analysis were done by an AI agent; there was no independent human re-analysis (README).

## 5. Proposed three-month fellowship investigation (research plan; not application text)

*[Section omitted from the public snapshot: a proposed follow-up research plan (private research-planning material). Retained in the private archive; see `PROVENANCE.md`.]*
