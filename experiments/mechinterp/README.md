# Mechanistic-Interpretability Pilot

This directory holds the pilot for `Qwen/Qwen3.5-9B` (revision `c2022362…`, bf16), run on the dataset in `../../data/synthetic/exploratory_v1` (864 items, 24 base scenarios). It has its own locked environment (`pyproject.toml`, `uv.lock`). Model weights and activations are not included.

The study is described in the top-level `README.md`, `METHODS.md` and `RESULTS.md`; commands are in `REPRODUCIBILITY.md`. Every result is a text-level authorization judgment; no tool was executed.

| Path | Contents |
|---|---|
| `src/mechinterp/model_io.py` | Model loading, prompt rendering, forced-choice scoring, residual-stream hooks |
| `src/mechinterp/label_diag.py` | Answer-interface diagnostic: item selection, prompt variants, scoring, parser |
| `configs/` | Frozen configurations: `pilot_config.json` (pilot) and `label_diag_config.json` (diagnostic) |
| `manifests/` | Model provenance (shard SHA-256s), chat template, model config, diagnostic items and prompts, `raw_results.sha256` |
| `scripts/smoke_test.py`, `run_full.py` | Behaviour labels and activation capture |
| `scripts/validate_extraction.py` | Activation integrity checks |
| `scripts/probes.py` | Task A / Task B probes, baselines and controls |
| `scripts/posthoc_*.py`, `audit_taskA_classes.py`, `label_diag_freegen_norms.py` | Post-hoc and audit analyses (labelled in their outputs) |
| `scripts/label_diag_run.py`, `label_diag_analyze.py` | Diagnostic runner and analysis |
| `scripts/verify_integrity.py` | Inference-free check of the preserved raw results (standard library only) |
| `results/` | Preserved raw outputs and analyses. `behaviour_full.jsonl` holds the original-interface scores for all 864 items. |
| `results/logs/` | Run logs; home-directory paths were replaced with `~` (see `../../PROVENANCE.md`) |
| `tests/` | Parser, metrics, scenario-disjoint CV, row-space projection, diagnostic selection, integrity |
