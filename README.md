# Authorization Guard Interpretability

An exploratory study of authorization reasoning, internal representation readout, and evaluation reliability in language models.

Language models used as authorization guards must distinguish between sufficient, missing, and contradictory evidence. Incorrect decisions can result in unauthorized permissions or unnecessary restrictions, particularly when these models are integrated into agentic systems.

This project investigates three questions:

- How reliably does a language model evaluate formally specified authorization policies?
- Can internal activations provide useful information about authorization errors?
- How sensitive are measured safety outcomes to the answer interface and scoring procedure?

The study uses 864 synthetic evaluation items derived from 24 underlying scenarios, a formal authorization oracle, and local bf16 inference with Qwen3.5-9B.

The experiments identify incorrect ALLOW judgments under contradictory evidence, strong internal decodability of conflict structure, and substantial sensitivity to answer scoring. However, activation probes do not demonstrate credible incremental error-prediction value beyond construction-aware baselines.

The repository contains the evaluation infrastructure, datasets, raw scoring results, analysis scripts, reproducibility instructions, and documented limitations.

This is exploratory research. It does not establish causal representation use, selective controllability, or generalization across models. No external tool execution was involved in the reported authorization experiments.

**Status:** a completed exploratory pilot. It has not been peer-reviewed.

## Reading guide

| To understand… | Read |
|---|---|
| The research problem and setup | This README, then [METHODS.md](METHODS.md) |
| The principal findings | [Principal findings](#principal-findings) below; [RESULTS.md](RESULTS.md) maps every number to its source file |
| The negative error-prediction result | [Negative result](#negative-result-error-prediction); [reports/46](reports/46_probe_and_baseline_results.md), [reports/47](reports/47_mechinterp_feasibility_decision.md) |
| The methodology | [METHODS.md](METHODS.md); detailed reports in [reports/](reports/) |
| How to reproduce the numbers | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| The limitations | [LIMITATIONS.md](LIMITATIONS.md) |
| Research provenance | [PROVENANCE.md](PROVENANCE.md) |
| Contributions and AI assistance | [Contributions and AI assistance](#contributions-and-ai-assistance) below |

## Setup

- **Data.** 864 items from 24 base scenarios. Each item gives a policy and a set of evidence lines. A three-valued oracle labels each item TRUE, FALSE, UNKNOWN or INCONSISTENT, which maps to the decision ALLOW, DENY, REQUEST_EVIDENCE or ESCALATE.
  - The primary set is 432 direct-wording items; the other 432 dual-wording items are an exploratory stress set.
- **Model.** [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, run in bf16 through Hugging Face `transformers`.
- **Behaviour.** Forced choice: the teacher-forced log-probability of each label.
- **Representations.** Residual-stream activations at 4 full-attention blocks and 2 positions.
- **Probes.** Linear probes under leave-one-scenario-out validation, compared against lexical, output-confidence and construction-aware baselines.
- **Interface diagnostic.** A final diagnostic re-scored 144 items (72 contradictory-evidence items and 72 controls) under four answer interfaces.

## Principal findings

| Finding | Value | Status |
|---|---|---|
| Original interface, 72 contradictory-evidence (CONFLICT) items | **0/72 correct escalations; 15/72 ALLOW**, 32/72 DENY, 25/72 REQUEST_EVIDENCE | Pre-declared measure |
| ESCALATE chosen on the full pilot | 0/864 items; max probability 0.043 | Descriptive |
| INCONSISTENT recall from activations (held-out scenarios) | 91.7% at block 7, mean-pooled; best 95.8% (block 15, mean-pooled); 77.8–95.8% across all 8 probes; TF-IDF 26.4% | Post-hoc, exploratory |
| Neutral letter identifiers, first-token scoring | at most **21/72** correct escalations; 0/72 unnecessary escalations on controls | Pre-declared sensitivity normalisation |
| Scoring sensitivity, mapping D | 35/72 under total-score scoring vs 19/72 under first-token scoring: a **22.2-percentage-point** difference | Post-hoc validity check |

Readout–behaviour discrepancy:
- Conflict structure is linearly decodable before the decision, yet under the original interface the model never escalates.
- Renaming ESCALATE to NEEDS_REVIEW leaves escalation at 0/72.
- Neutral letter identifiers raise it only partly, and introduce other errors on controls: 2 and 1 controls become wrong ALLOW under first-token scoring.

On the decodability claim:
- The decodable property **tracks the presence of contradictory evidence lines in the input**. It is not shown to be an abstract representation of logical inconsistency.
- Early-layer decodability shows information that bag-of-n-grams features miss. It may still reflect the surface configuration of the evidence, and it does not imply reasoning about the policy.

On the scoring result:
- Total-score forced choice for mapping D failed the free-generation agreement check (8/12).
- The first-token figure agrees with generation (12/12) and is the better-supported sensitivity result. It is not an independently validated replacement scoring procedure.

## Negative result: error prediction

The pre-declared question was whether activations predict the model's authorization errors beyond cheaper signals. The base rate is 0.470, over the 432 primary items.

| Score | AUROC [95% CI] |
|---|---|
| Text (TF-IDF) + output margin | 0.838 [0.822, 0.854] |
| Best activation probe (block 31 pre-norm, last token) | 0.919 [0.905, 0.935] |
| Construction-aware structured features | 0.955 [0.938, 0.973] |
| Construction-aware structured features + margin | 0.969 [0.957, 0.981] |

**Conclusion:** activation probes did not show credible independent predictive value beyond item structure.
- They beat text + margin, but not the structured baselines.
- They show no error-ranking signal within cells where the structure is fixed.

The structured baselines receive dataset-construction information. They are diagnostic ceilings, not deployable predictors.

## Scope distinctions

- **Judgments, not actions.** All decisions are text-level authorization judgments. No tool, API or resource access was executed or simulated. An incorrect ALLOW label is not a demonstrated unauthorized execution.
- **Decodability, not causal use or controllability.** No steering, patching or other intervention was performed.
- **Exploratory, not confirmatory.** The data are templated and were inspected during development. The 24 scenarios, not the 864 items, are the independent units.
- **Forced choice, not free generation.** Decisions are scored by forced choice; free generation was checked on 60 items for the original interface and on 12 items per diagnostic condition.
- **bf16 Hugging Face, not quantized Ollama.** All labels here come from the bf16 checkpoint. An earlier behavioural study with Q4_K_M-quantized models served by Ollama is not part of this repository, and its labels are not used.
- **Known effects.** Sensitivity of multiple-choice evaluations to label identity, option order and scoring normalisation is established in prior work; see [LIMITATIONS.md](LIMITATIONS.md). The diagnostic here measures it for this task; it does not claim the effect as new.

## Repository layout

```
data/synthetic/exploratory_v1/   864 items + system prompt (frozen; SHA-256 in manifests/)
src/authorization_closure/       item generator and three-valued oracle (standard library only)
tests/                           oracle and generator tests (60)
experiments/mechinterp/          pilot: locked environment, code, configs, manifests, raw results, tests (9)
reports/                         detailed reports 43–49 (archival, with marked omissions)
research/                        decision-log excerpt: pre-declarations, amendments, post-hoc labels
```

## Quick verification (no model; CPU only)

```
cd experiments/mechinterp
shasum -a 256 -c manifests/raw_results.sha256
python3 scripts/verify_integrity.py      # recomputes the headline behaviour counts from raw log-probs
```

The tests and full regeneration are described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). This snapshot has 69 tests: 60 at the root and 9 in `experiments/mechinterp`.

## Contributions and AI assistance

This project was conducted as an independent, AI-assisted research investigation.

My contributions included defining the research objectives, directing the experimental workflow, reviewing methodological decisions, evaluating competing hypotheses, and determining the scope of subsequent investigations, as well as participating in experimental execution, analysis, and debugging.

AI coding and research assistants were used for implementation, literature exploration, documentation.

The repository preserves experimental protocols, negative findings, raw results, integrity checks, and documented methodological limitations to make the investigation transparent and reproducible.

The findings have not undergone independent peer review or external replication.

**Historical disclosure (from the research archive).** The archive records that the experimental design, code, analyses and reports were produced by an AI agent (Anthropic's Claude, in Claude Code) under the author's direction. The author set the objectives, constraints and approval checkpoints. No human independently re-ran or re-analysed the results, and no human annotation was performed. The archival reports and decision log in this repository were written by that agent.

## License

| Material | License |
|---|---|
| Source code | Apache-2.0 ([LICENSE](LICENSE)) |
| Documentation, reports, the original synthetic dataset and result files | CC BY 4.0 ([LICENSE-docs-data.md](LICENSE-docs-data.md)) |
| Third-party files (model chat template and config from `Qwen/Qwen3.5-9B`) | their original Apache-2.0 licence; see [NOTICE](NOTICE) |

Model weights are not included. Download them from the official model repository at the pinned revision.
