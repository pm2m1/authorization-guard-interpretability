# 49 — Final Mechinterp Claims and Limitations

This is the closing statement for the Qwen3.5-9B pilot (reports 43–48) and the last pre-application experiment. It is exploratory throughout. Nothing is pushed or published.

## 1. What the evidence supports

| # | Claim | Strength | Source |
|---|---|---|---|
| 1 | Local bf16 inference, hooks and probing of Qwen3.5-9B on a 48 GB Mac are **technically feasible** and bitwise reproducible | Strong (engineering) | 43, 45 |
| 2 | **Error prediction: no credible added value beyond item structure.** Activations beat text + margin (best AUROC 0.919 vs 0.838) but not structured features (0.955 / 0.969), and they show no within-cell signal. **This negative conclusion stands.** | Strong for this data set | 46, 47 |
| 3 | Under the original interface the model **never selects ESCALATE** (0/864; max p 0.043). Renaming it NEEDS_REVIEW does not change this (0/72). | Strong | 44, 48 |
| 4 | **Output-interface sensitivity.** Neutral identifiers with explicit mapping raise *generation-validated* correct escalation to 17–21/72 (mapping C) and 19/72 (mapping D, first-token), with 0 unnecessary escalations on 72 controls. The D total-score figure (35/72) is a termination artefact. | Moderate: two mappings, 24 clusters | 48 |
| 5 | **Readout–behaviour discrepancy persists across the four tested interfaces.** INCONSISTENT is linearly decodable pre-decision, with recall 0.78–0.96 on held-out scenarios. Behaviour escalates on at most about 29% of conflict items. Justifications often name the contradiction without escalating (11 of the 13 that mention it). | Moderate, exploratory; per-class analysis post-hoc | 46, 48 |

## 2. Required narrowing of the representation claim

- **INCONSISTENT ⇔ the CONFLICT construction.** The construction-aware structured baseline identifies it perfectly, but only through the condition label. Non-condition metadata does not (F1 ≤ 0.18), and nor does TF-IDF (F1 0.28).
- What the activations encode is therefore a property that **tracks the presence of contradictory evidence lines in the input**. The claim is "decodable from internal states better than from bag-of-n-grams". It is **not** "the model represents logical inconsistency" in any sense separable from detecting that surface configuration.

## 3. What is *not* claimed

- **No causal use.** Decodability does not show that the model uses the direction.
- **No controllability.** No activation intervention was approved or run.
- **Lexical preference is not the sole cause.** Renaming alone did nothing; the effective change bundled identifier, list position, mapping text and response format.
- **Label bias in general is not ruled out.** Only four interfaces were tested, and content-free calibration was not tested.
- **No generalisation** beyond this model, prompt and templated data set.
- **No tool execution.** *(Housekeeping addendum, D47.)* Every decision in reports 43–49 is the model's text-level authorization judgment on a synthetic scenario: label scores or a short generation. No tool, API or resource access was executed or simulated, and no agent loop was run. "Fail-open" and "wrong ALLOW" mean the ALLOW label was chosen; nothing was actually permitted.

## 4. Limitations (consolidated)

- **Data.** 864 synthetic templated items (24 scenarios), previously inspected. The diagnostic used 144 of them. Oracle labels follow the construction.
- **Model and interface.** One checkpoint, one system-prompt family, forced-choice scoring validated against greedy generation on 60 + 48 items. One normalisation failure was found (mapping D, post-hoc).
- **Representations.** 4 full-attention blocks × 2 positions, linear probes only. No linear-attention layers, heads or SAE features. No activations were captured under the alternative interfaces.
- **Statistics.** 24-cluster bootstrap. Small within-cell samples (n = 24). The two neutral mappings each ran once.
- **Process.** Amendments AM1 and AM2, and the post-hoc analyses D43 and D46, are disclosed. The analysis was done by an AI agent, with no independent human re-analysis.

## 5. Implications for the proposed three-month investigation (reports/47 §5)

*[Section omitted from the public snapshot: implications for a proposed follow-up research plan (private research-planning material). Retained in the private archive; see `PROVENANCE.md`.]*
