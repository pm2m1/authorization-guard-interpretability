# Limitations

## Data

- There are 864 synthetic, templated items from 24 author-written base scenarios. Oracle labels follow the construction, and no human validated the policy semantics.
- The items were inspected during earlier development, so this is not a held-out confirmatory set. Nothing was preregistered externally; pre-declarations were frozen internally in Git before each run.
- The dual-wording stress set contains contestable oracle labels. It is reported separately and is not primary.

## Model and interface

- One checkpoint (`Qwen/Qwen3.5-9B`, bf16), one system-prompt family and one runtime (transformers on Apple MPS, with pure-PyTorch kernels for the linear-attention layers).
- Forced-choice scoring was validated against greedy generation on 60 items for the original interface, and on 12 items per condition in the interface diagnostic. Total-score normalisation failed that check for mapping D (found post-hoc).
- Only four answer interfaces were tested. Content-free label calibration was not examined, so label bias in general is not ruled out.
- Renaming alone had no effect. The effective neutral-letter change bundled identifier, list position, explicit mapping text and response format, which cannot be separated here.
- Only two neutral-identifier mappings were tested, so token identity is not fully separated from option position.
- Decisions come from forced-choice scoring, not free generation. Under mapping D, total-score scoring failed the generation-agreement check. First-token scoring agrees with generation on the 12 checked items, but that is a sensitivity result, not a validated replacement procedure.
- The tested label and scoring effects do not explain all escalation failures. Escalation stays at or below 21/72 under every tested interface, and the letter interfaces introduce other control errors.
- **Prior work.** Sensitivity of LLM multiple-choice evaluation to label tokens, option order and answer scoring is established, for example:
  - Zhao et al., "Calibrate Before Use: Improving Few-Shot Performance of Language Models" (ICML 2021);
  - Pezeshkpour and Hruschka, "Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions" (2023);
  - Zheng et al., "Large Language Models Are Not Robust Multiple Choice Selectors" (ICLR 2024).

  This repository measures the effect for one authorization task and claims no novelty for it.

## Representations and probes

- Activations were taken at 4 full-attention blocks and 2 token positions. The 24 linear-attention layers, attention heads and SAE features were not examined.
- Only linear probes were used. Block 31 is captured before the final norm. No activations were recorded under the alternative interfaces.
- The frozen configuration specified float32 activation storage, but the saved activations are float16 (disclosed in reports/45 §3).
- Early-layer probes (for example, block 7 mean-pooled) use contextual representations, not raw lexical features. Beating TF-IDF does not show that the decoded property goes beyond the surface configuration of the evidence lines.
- The class-level (per-class recall) analyses are post-hoc.
- The structured baseline encodes the dataset construction exactly, including the condition label that defines CONFLICT. It is a diagnostic ceiling, not a deployable predictor.
- Errors are largely determined by the cell: ESCALATE is never chosen, and the cumulative family fails almost entirely. "Error prediction" is therefore largely cell prediction, and only two cells allow a within-cell comparison (n = 24 each).

## Inference

- Decodability is correlational. No steering, patching or ablation was run, so causal use and selective controllability are untested.
- The decodable property tracks the CONFLICT construction, meaning contradictory evidence lines about the same fact. It is not shown to be an abstract representation of logical inconsistency.
- Intervals come from a bootstrap over only 24 scenario clusters, so they are approximate. Each neutral mapping was run once.
- The 864 items are repeated evaluations built from 24 scenarios. The scenarios, not the items, are the independent units, and all validation and intervals are grouped by scenario.
- An incorrect ALLOW label is a text-level judgment. No tool or action was executed, so no unauthorized execution was demonstrated.
- The work has had no peer review and no independent human replication.

## Process

- The design, code, analyses and reports were produced by an AI agent under the author's direction. There was no independent human re-analysis.
- Two amendments (AM1, AM2) and two sets of post-hoc analyses (D43, D46) are disclosed in `research/decision_log_mechinterp.md`. AM2 changed the computation, but not the estimator, after one intermediate result had been seen.
