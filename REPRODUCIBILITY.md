# Reproducibility

There are four levels, from cheapest to most expensive. Levels 0–2 need no model weights and run on CPU.

| Level | What it checks | Needs |
|---|---|---|
| 0 | Raw-result checksums; recomputes headline behaviour counts | Python ≥ 3.10, standard library only |
| 1 | Unit, property-based and integrity tests; lint | `uv`; Python 3.12 |
| 2 | Regenerates the dataset byte-for-byte | `uv`; Python 3.12 |
| 3 | Regenerates the behaviour labels, activations, probes and diagnostic | Model weights (about 19.3 GB), Apple Silicon with MPS; tested with 48 GB of unified memory (peak MPS allocation 25.6 GB) |

## Environments

There are two separate locked environments:
- the **repository root** (`pyproject.toml`, `uv.lock`) is the dataset generator and oracle. It has no runtime dependencies; the dev tools are pytest, hypothesis and ruff.
- **`experiments/mechinterp/`** has its own `pyproject.toml` and `uv.lock`: torch 2.15.0, transformers 5.22.2, scikit-learn 1.9.1, accelerate < 1.16.

Both use Python 3.12, which `.python-version` pins.

## Level 0: integrity (no dependencies)

```
cd experiments/mechinterp
shasum -a 256 -c manifests/raw_results.sha256
python3 scripts/verify_integrity.py
```

`verify_integrity.py` reads only preserved files. It checks the following:
- the 864 unique item IDs;
- the stored decision equals the argmax of the stored log-probs (864/864);
- the 144 diagnostic items match the raw scores by `item_id`, and the source prompts by SHA-256;
- recomputed from the raw log-probs: **0/72** correct escalations and **15/72** ALLOW on CONFLICT items, **0/72** unnecessary escalations on controls, and 0/864 ESCALATE decisions overall;
- the 8-item replay is bitwise equal.

It exits with status 1 if any check fails.

## Level 1: tests and lint

```
uv sync && uv run pytest -q && uv run ruff check .                      # repository root
cd experiments/mechinterp && uv sync && uv run pytest -q tests && uv run ruff check .
```

`uv sync` in `experiments/mechinterp` installs torch and transformers, but the tests never load the model.

## Level 2: dataset regeneration

```
uv run python -m authorization_closure.generate --out /tmp/regen --seed 20261125
shasum -a 256 /tmp/regen/items.jsonl /tmp/regen/system_prompt.txt
# expected: items 9d9ec3f7…2d12, system prompt 9634bb0b…8711 (manifests/exploratory_v1_dataset.json)
```

## Level 3: full regeneration (requires model weights)

**Work in a fresh clone.** The run scripts write over the preserved results:
- `run_full.py dev` rewrites `results/dev_agreement.jsonl`;
- `run_full.py full` **resumes**: it skips every item already in `results/behaviour_full.jsonl`, so with the preserved file in place it regenerates **no** activations;
- `validate_extraction.py` rewrites `results/extraction_validation.json`, which holds the original activation checksums.

1. **Download and verify the weights.** About 19.3 GB, Apache-2.0, ungated.
   ```
   cd experiments/mechinterp && uv sync
   uv run python -c "from huggingface_hub import snapshot_download as d; print(d('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'))"
   ```
   Compare each `model.safetensors-*` SHA-256 with `manifests/model_provenance.json`. From here on, set `HF_HUB_OFFLINE=1`.
2. **Behaviour and activations.** About 40 min on an M-series Mac. Keep the machine awake, for example with `caffeinate -i`.
   ```
   git mv results/behaviour_full.jsonl results/behaviour_full.preserved.jsonl    # or move it outside the clone
   HF_HUB_OFFLINE=1 uv run python scripts/smoke_test.py
   HF_HUB_OFFLINE=1 uv run python scripts/run_full.py dev
   HF_HUB_OFFLINE=1 uv run python scripts/run_full.py full                      # writes data/activations/reps_00..08.npy
   uv run python scripts/validate_extraction.py
   ```
3. **Probes.** About 4.7 h on 15 CPU cores with permutations, and 5–10 min without.
   ```
   uv run python scripts/probes.py --perms-b 100 --perms-a 20 > results/probe_results.json
   uv run python scripts/probes.py --perms-b 0 --perms-a 0 --save-oof results/oof_scores.json > results/probe_results_oof_rerun.json
   uv run python scripts/probes.py --perms-b 0 --perms-a 0 --exclude-smoke --save-oof results/oof_scores_excl_smoke.json > results/probe_results_excl_smoke.json
   uv run python scripts/posthoc_within_cell.py results/oof_scores.json > results/posthoc_within_cell.json        # post-hoc
   uv run python scripts/posthoc_taskA_per_class.py > results/posthoc_taskA_per_class.json                        # post-hoc
   ```
4. **Answer-interface diagnostic.** About 1,728 scoring passes plus 48 generations.
   ```
   HF_HUB_OFFLINE=1 uv run python scripts/label_diag_run.py replay
   HF_HUB_OFFLINE=1 uv run python scripts/label_diag_run.py primary
   HF_HUB_OFFLINE=1 uv run python scripts/label_diag_run.py freegen
   uv run python scripts/label_diag_analyze.py > results/label_diag/analysis.json
   uv run python scripts/audit_taskA_classes.py > results/label_diag/audit_taskA.json
   uv run python scripts/label_diag_freegen_norms.py > results/label_diag/freegen_by_norm.json                     # post-hoc
   ```

**What to expect.**
- On the original machine (Apple M-series, MPS, the locked versions above), label log-probs were bitwise identical across separate processes.
- On other hardware or kernels, small numerical differences are expected. The `created_utc` and `forward_seconds` fields always differ.
- Compare scores and decisions, not file hashes.
- The activations (54 MB) are not distributed. The original shard SHA-256s are in `results/extraction_validation.json`.
