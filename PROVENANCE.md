# Provenance

This repository is a **curated public snapshot derived from a private research archive**. It is not the complete archive.

| | |
|---|---|
| Source archive | a private local Git repository, 48 commits, 2026-11-25 to 2026-11-27. It is preserved unchanged, and every excluded item below is retained there. |
| Source commit | `e6699e6` |
| Creation procedure | 1. Extract the selected paths with `git archive e6699e6`, so files are the committed bytes. 2. Apply the redactions and omissions listed below. 3. Add the public documentation. 4. Run the checks listed at the end. 5. Commit everything as one initial commit in a new repository. |
| History | the snapshot starts with a single new commit; the archive's commit history is not included |
| File-level record | [`provenance_manifest.tsv`](provenance_manifest.tsv): every file, its status (verbatim / modified / excerpt / new) and its archive and snapshot SHA-256 |

## Verbatim files (73)

- The dataset and its freeze manifest.
- The generator and oracle modules: `__init__`, `base_scenarios`, `generate`, `kleene`, `oracle`, `serialization`.
- 3 of their 4 test files.
- All code, configurations, manifests and result files of `experiments/mechinterp/`, except the 8 files listed below.
- Reports 44, 45, 46 and 48.
- `.python-version`.

## Modified files (16) and the reason for each

| File | Change | Reason |
|---|---|---|
| `experiments/mechinterp/results/smoke_test.json` | **Redaction:** the absolute home-directory prefix in one warning string replaced by `~` | privacy (local username); no result value changed |
| `experiments/mechinterp/results/logs/{download,probes,probes_aborted_run1,smoke_test}.log` | **Redaction:** the same substitution (1, 28, 29 and 1 occurrences) | privacy |
| `experiments/mechinterp/manifests/raw_results.sha256` | the `smoke_test.json` hash updated to the redacted file; the other 9 lines identical | consistency after redaction |
| `reports/43…` | §1 replaced by a marked omission | archive housekeeping, plus an advisory table for an unrelated, excluded study |
| `reports/47…`, `reports/49…` | §5 replaced by a marked omission | proposed follow-up research plan; private research-planning material |
| `pyproject.toml`, `uv.lock` (root) | project renamed; scope reduced to the dataset generator; unused runtime dependencies and the stale Ollama/Docker marker removed; re-locked offline with **no version changes** among the remaining packages. A ruff line-length exemption was later added for `scripts/generate_figures.py`; the lock is unaffected. | the excluded code needed those dependencies; long caption strings in the figure script |
| `tests/unit/test_oracle.py` | one line: `HealthCheck.filter_too_much` added to the suppressed Hypothesis health checks of the shared settings `S`; no assertion, strategy or example count changed | the property test `test_p1_p2_monotone_under_lcwa_when_added_fact_is_on_open_predicate` filters inputs by design (`assume`), and the health check made it fail intermittently (1 of 17 and 1 of 7 runs during preparation). After the change: 0 failures in 40 runs. The archive copy keeps the original line. |
| `experiments/mechinterp/pyproject.toml` | the `description` string only; `uv.lock` unaffected | it referred to archive-internal investigations |
| `README.md`, `experiments/mechinterp/README.md`, `.gitignore` | replaced by public versions | the archive versions describe the whole archive and its workflow |

The archive SHA-256 of each modified file is listed in `provenance_manifest.tsv`.

## Excerpted file (1)

`research/decision_log_mechinterp.md` contains entries D40–D47 of the archive's append-only decision log, verbatim, under a short header. Entries D1–D39 concern excluded investigations.

## New files (19)

- **Documentation:** `LICENSE` (standard Apache-2.0 text), `LICENSE-docs-data.md`, `NOTICE`, `CITATION.cff`, `RESULTS.md`, `METHODS.md`, `REPRODUCIBILITY.md`, `LIMITATIONS.md`, `PROVENANCE.md`, `reports/README.md` and `provenance_manifest.tsv`.
- **Presentation update (second commit):**
  - `scripts/generate_figures.py`, a standard-library figure generator that reads the preserved results;
  - six SVGs in `figures/`;
  - `.github/workflows/ci.yml`, CPU-only checks.
- **Final presentation update (third commit):**
  - native Mermaid architecture diagrams in `README.md` (research pipeline) and `METHODS.md` (model architecture and activation extraction), derived from the implementation and the two SVG diagrams, which remain as detailed exports;
  - the README example is described as the highest-margin incorrect ALLOW case, and its probabilities as normalised candidate probabilities under forced-choice scoring;
  - the AI-assistance sentence lists experimental design, implementation, literature exploration, execution, analysis, debugging and documentation.
- **CI follow-up:**
  - the pilot-test job moved from the retired `macos-14` runner image to `macos-15`; its steps are unchanged;
  - after the first fully successful run (Actions run 46008437704: 60 + 9 tests passed, 10/10 checksums, integrity PASS, ruff clean), a CI status badge was added to the README.

- These summarise, index and visualise archive material and contain no new experimental results. Every number in them was checked against the preserved result files. The figures are generated from those files, and the README example is a recorded item quoted verbatim.
- The overview and the contribution statement in `README.md` were provided by the author. The historical AI-assistance disclosure from the archive is retained alongside them.

## Excluded (retained in the archive)

| Archive material | Reason |
|---|---|
| An earlier behavioural study of authorization guards on the same dataset: its reports, `data/raw/` model outputs, `data/diagnostic/`, `results/`, `scripts/`, the Ollama client and runners | A separate, closed investigation using **Q4_K_M-quantized models through Ollama**: a different runtime, quantization and interface. None of its labels are used here. |
| A separate study of MCP-server behaviour drift: `src/mcp_drift/`, `docker/`, `data/raw/mcp_runs/`, its reports | Unrelated to this study. Its sandboxed harness observed behaviours matching **two already-published GitHub security advisories**, one used deliberately as a positive control. These are reproductions of known vulnerabilities, not discoveries. No reproduction material is included. |
| Research-direction and novelty reviews, and the `literature/` PDFs | Private research-planning material; supplied input documents whose authorship and redistribution rights are not established |
| Other `research/` files: protocols, human-annotation packages, prior-art matrix, decision-log entries D1–D39 | They belong to the excluded investigations |
| Archive commit history | private working history |
| Activations (`experiments/mechinterp/data/activations/`, 54 MB) | size. They can be regenerated (REPRODUCIBILITY.md, Level 3), and the original SHA-256s are in `results/extraction_validation.json`. |
| Model weights (about 19.3 GB) | third-party; download from the official repository at the pinned revision (shard SHA-256s in `manifests/model_provenance.json`) |

## Verification of this snapshot

These were run on CPU only, without loading the model or running experiments:

| Check | Result |
|---|---|
| `shasum -a 256 -c experiments/mechinterp/manifests/raw_results.sha256` | 10/10 OK |
| `python3 experiments/mechinterp/scripts/verify_integrity.py` | PASS: 864 decisions equal their top-scoring label; 144/144 diagnostic IDs and prompt hashes matched; 0/72 escalations, 15/72 ALLOW, 0/72 unnecessary escalations |
| Dataset regeneration (`generate --seed 20261125`) | byte-identical `items.jsonl` and `system_prompt.txt` |
| Lock files (`uv lock --check`, both environments) | valid |
| Root tests / ruff | 60 passed in each of 40 consecutive runs / clean |
| Figure generation | deterministic: two runs give byte-identical SVGs; every plotted value matches the source files |
| `experiments/mechinterp` tests / ruff | 9 passed / clean (fresh environment from `uv.lock`, offline) |

The full archive's test suite has 106 tests (97 + 9), because it also covers the excluded code.
