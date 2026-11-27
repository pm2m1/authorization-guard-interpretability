# 43 — Mechanistic-Interpretability Pilot: Hygiene, Methods Requirements, Environment and Model

**Status: STOPPED FOR APPROVAL before any model-weight download** (§5). This is a separate exploratory investigation. The earlier investigations stay closed. AI assistance: designed and written by an AI agent (Claude in Claude Code) under your direction (README disclosure).

## 1. Repository hygiene (commit `c298607`, decision log D39)

*[Section omitted from the public snapshot. It recorded housekeeping of the private research archive (date corrections, README restructuring) and the advisory mapping for a separate, archived MCP-server investigation. It does not bear on the pilot. Retained in the archive; see `PROVENANCE.md`.]*

## 2. Methodological requirements from the motivating papers

These were read through fetch agents, verbatim unless marked, and only the methods were extracted.

**Mentor paper** (*Multimodal Safety Evaluation Should Measure Controllability Beyond Classification*, arXiv 2610.06452; authors include Hanwool Lee, Dasol Choi and Haon Park of AIM Intelligence):
- It proposes a reporting profile, not a score. The four axes are detectability, cross-modal specificity, intervention sensitivity and benign-preserving selectivity. "Predictiveness alone does not show that manipulating the representation will change the target signal or behavior" (§3.2).
- **Models:** LlavaGuard-v1.2-7B-OV and **Qwen3.5-9B**. **Representation:** mean-pooled **layer-24** hidden states (D_h = 4096).
- **Probes:** a 3-layer MLP (4096 → 1024 → 256 → 3) on pooled states, plus a "proxy-residual" probe. Splits: one random and one group-holdout (≈ 7:3; the grouping key is unnamed). No k-fold, no CIs, no shuffled-label or lexical baselines.
- **SAE (App. C.2), confirmed:** "We train a TopK Sparse Autoencoder (TopK-SAE) on the mean-pooled layer-24 hidden states using the training split". **16,384 latents, K = 128**, AdamW, 10k steps, batch 32. This is task-specific, trained by the authors. No reconstruction or dead-latent metrics are reported. Interventions: deletion, insertion, clamping and runtime projection at layer 24.
- **Requirement for us:** declare the representation, probe and operating range in advance, and keep decodability separate from intervention claims.

**SAKIKO** (2609.36138):
- **Readout:** L2 logistic regression on the residual stream at the **last prompt position** (pre-decision), standardised, on a fixed train/val/test split.
- **Scoring:** teacher-forced scoring of complete candidate decisions.
- **Steering:** a direction estimated separately from the probe, with **59 budget-matched random directions** as a specificity null.
- "probe decodability does not guarantee steerability".
- It **disqualified Qwen3.5-9B** before intervention for class skew and too few baseline-correct rows (29 < 30). That is a directly relevant feasibility warning.

**Verdict Without the Rule** (2610.12313):
- Last-token, training-free mean-difference direction. The anchor layer is selected on a split disjoint from direction fitting.
- An empirical null from permutations; rule deletion, swap and negation.
- No output-confidence baseline. Steering gave no significant gain.

**Auditing the Judge** (2611.14323):
- **Error prediction** from internal features (per-head attention outputs at the final prompt position) vs the judge's own **p(yes)**.
- Metrics: AUROC, AP and **error recall at 5/10/20% review**, with a paired class-stratified bootstrap. Surface-feature residualisation.
- Its stated gaps: no shuffled labels, no grouped CV, and an oversampling leakage risk. "A fairer access comparison would train the same classifier and split on p(yes), shallow statistics…"
- Transfer loses AUROC (−0.040 to −0.270).

**TACIT** (2609.33039):
- L2 logistic regression on a frozen backbone, with layer and pooling chosen by validation **inside training folds**.
- **Group-aware splits** (same scenario on one side), standardisation fitted on training data only, macro-F1, bootstrap CIs, and a placebo-edit control.
- Output confidence is not tested as an incremental baseline.

**Requirements this pilot adopts:**
- pre-decision readout;
- linear L2 probes;
- grouped CV by the 24 base scenarios, with all selection and scaling done inside folds;
- shuffled-label controls;
- every layer reported;
- error prediction compared against output margin, structured, TF-IDF and combined baselines;
- AUROC, AP, recall at fixed review budgets and risk–coverage curves;
- scenario-cluster bootstrap intervals;
- **no causal claims.**

## 3. Model: `Qwen/Qwen3.5-9B` (verified from the HF API and raw config files only; no weights)

| Item | Value |
|---|---|
| Revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (last modified 2026-03-02) |
| Licence / gating | Apache-2.0; not gated |
| Type | Post-trained (`base_model: Qwen/Qwen3.5-9B-Base`); pipeline `image-text-to-text`; architecture `Qwen3_5ForConditionalGeneration` (includes a 27-layer vision tower) |
| Size | 9,653,104,368 parameters (BF16); **4 safetensors shards, 19.33 GB in total** (5.28 + 5.34 + 5.37 + 3.33 GB; LFS sha256 prefixes db6f444b…, 31c7d7e2…, 7ec36ba3…, b62b0c4c…) |
| Text stack | **32 decoder layers, hidden size 4096. Hybrid attention:** `layer_types` puts `full_attention` every 4th layer (0-indexed 3, 7, …, 31) and **linear (gated-delta-rule) attention** elsewhere. GQA 16 heads / 4 KV heads, head_dim 256, vocabulary 248,320. Also has an MTP head. |
| Chat template | `chat_template.jinja` (sha256 `a4aee8af…`, saved in `experiments/mechinterp/manifests/`). With `enable_thinking=False`, the generation prompt is `<|im_start|>assistant\n<think>\n\n</think>\n\n`. |
| Config | sha256 `d0883072…` (saved). `transformers_version` 4.57.0.dev0 at authoring. |
| Ollama comparison | The earlier exploratory runs used Ollama `qwen3.5:9b-q4_K_M` (Q4_K_M quantised, a different runtime). **Its behaviour labels are not reused.** |

**Feasibility risk specific to this architecture.** The 24 linear-attention layers use `chunk_gated_delta_rule` and causal-conv1d kernels. On Apple MPS, with neither `fla` nor `causal_conv1d` nor the HF `kernels` package installed, transformers uses its **pure-PyTorch fallback**. This was verified in the installed source: `use_kernel_func_from_hub_with_fallback` → "Torch only path". Speed and MPS op coverage for that path are **unknown until tested**. HF Hub kernels are used only "if requested", and none will be requested.

## 4. Environment (separate, locked)

`experiments/mechinterp/` has its own `pyproject.toml` and `uv.lock` (75 packages):
- Python 3.12.14;
- **torch 2.15.0** (released 2026-10-28), MPS available and built;
- **transformers 5.22.2** (2026-11-06); `qwen3_5` module and `Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM` import OK;
- accelerate < 1.16;
- scikit-learn 1.9.1;
- numpy, pandas, scipy.

Pinned versions are at least two weeks old. No interpretability libraries (native PyTorch hooks only). Disk: 604 GiB free.

## 5. Pre-declared plan for the smoke test (≤ 12 existing items; to run only after approval)

1. **Download.** `huggingface_hub.snapshot_download("Qwen/Qwen3.5-9B", revision=c2022362…)` into `~/.cache/huggingface`. That is outside the repo, about 19.3 GB, and the exit status will be checked. Then verify each shard's sha256 against its LFS OID. After that, all runs use `HF_HUB_OFFLINE=1`.
2. **Load in bf16 on MPS.** Record load time, peak memory (`torch.mps.driver_allocated_memory()` plus process RSS) and any CPU fallback warnings. The text stack is used; the vision tower is loaded but never fed images.
3. **Pre-declared extraction layers.** `hidden_states` indices **8, 16, 24, 32**: the outputs of decoder blocks 7, 15, 23 and 31, all full-attention blocks. Index 24 matches the mentor paper's layer 24 (convention to be confirmed against `output_hidden_states`).
   - **Primary representation:** the residual stream at the **last token of the generation prompt** (after `<think>\n\n</think>\n\n`), before any answer token.
   - **Secondary, pre-declared now:** the mean over user-message tokens.
4. **Hook checks:**
   - forward hooks on the 4 blocks capture shapes [1, T, 4096];
   - hooked and unhooked runs must give **bitwise-identical logits** (no modification);
   - the hook output must equal `output_hidden_states`.
5. **Scoring.** Teacher-forced log-probability of each complete label string (ALLOW, DENY, REQUEST_EVIDENCE, ESCALATE), tokenised after the generation prompt.
   - The decision is the argmax of the total label log-probability; the margin is top minus second.
   - The label tokenisations are recorded, including multi-token labels.
   - This is a **forced-choice interface, distinct from the earlier free-generation behaviour.**
   - Agreement with greedy generation (≤ 80 new tokens, pilot parser) is checked on a pre-declared development subset.
6. **Reproducibility and throughput.** Two identical forward passes per item (compare logits). Forward passes per second, tokens per second, total wall time.
7. **Items.** 12 existing exploratory items: 1 base (DA01) × {existential, universal} × {POS, NEG, MISS} × {direct, dual}. Used for **technical checks only**.

## 6. Approval required

**Download of `Qwen/Qwen3.5-9B` revision `c2022362…` (≈ 19.3 GB, Apache-2.0)** into the local Hugging Face cache. Nothing further will run until you approve. If 9B turns out to be impractical on MPS, I will report the blocker and ask again before considering a smaller model.

---

## 7. Download and smoke-test results (added after your approval; `experiments/mechinterp/results/smoke_test.json`)

**Download.** Exit status 0; about 7 minutes; 19.3 GB to `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c2022362…`. **All 4 shard SHA-256s equal the published LFS OIDs** (`manifests/model_provenance.json`). All runs afterwards used `HF_HUB_OFFLINE=1`.

| Check | Result |
|---|---|
| Loading | OK in bf16 on MPS, 12 s. **No `PYTORCH_ENABLE_MPS_FALLBACK`** was needed: no unsupported-operation errors. |
| Kernel fallbacks | Transformers warns that `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`, `causal_conv1d_fn/_update` use "reference PyTorch implementation … correct but much slower" (expected) |
| Memory | 18.85 GB after load. **Peak MPS driver allocation 22.44 GB** (of 48 GB) |
| Decoder path | `model.language_model.layers`, 32 blocks. Blocks 7, 15, 23 and 31 are `Qwen3_5Attention` (full attention), as pre-declared. |
| Hooks | Shapes [1, T, 4096]. **Logits are bitwise identical with and without hooks.** Hook output equals `hidden_states[b+1]` exactly for blocks 7, 15 and 23. For **block 31 it differs** (max diff 200), because HF's final `hidden_states[32]` comes **after the final RMSNorm**. The pilot therefore records block 31 **pre-norm**, documented in the frozen config. |
| Label tokenisation | ALLOW [51535, 198]; DENY [88357, 56, 198]; REQUEST_EVIDENCE [10477, 2021, 7248, 9848, 198]; ESCALATE [1515, 47868, 2260, 198]. Multi-token labels with distinct first tokens. |
| Forced choice vs greedy generation | **12/12 agreement.** All generations parsed STRICT. Full-label vs first-token-only decision: 11/12. |
| Reproducibility | Label log-probs **bitwise identical** across two runs on 4 items |
| Throughput | Mean forward pass 0.76 s (prompts of about 360–376 tokens); maximum 3.78 s, the first, warm-up call. About 7.1 s per item including 4 scoring passes and an 80-token generation. |
| Behaviour (12 items, DA01; technical only) | 9/12 correct. Errors: ∃-POS dual → DENY; **∀-MISS dual → ALLOW** (fail-open); ∃-MISS dual → DENY. All 3 are dual wordings. |

**Projected full run.** 864 items × 4 scoring passes ≈ 864 × 3 s ≈ **45 minutes**, plus generation on a 48-item development subset (≈ 6 min). **Feasible on this hardware.**

**Full-run configuration frozen** in `experiments/mechinterp/configs/pilot_config.json`, before any full-run result:
- primary analysis = **direct wording (432 items)**; dual wording = exploratory stress cases (`reports/14` ambiguities);
- leave-one-scenario-out outer CV with grouped inner CV;
- shuffled-label controls;
- the baseline set;
- stopping rules: fewer than 30 primary-set errors; generation agreement below 90%.
