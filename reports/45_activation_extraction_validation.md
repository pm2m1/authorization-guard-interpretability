# 45 — Mechinterp Pilot: Activation Extraction Validation

Source: `experiments/mechinterp/results/extraction_validation.json` and `results/smoke_test.json`. Activations are stored at `experiments/mechinterp/data/activations/reps_00…08.npy`: 54 MB, float16, shape [N, 4 blocks, 2 representations, 4096]. They are gitignored because of size, and the per-shard SHA-256s are recorded in the validation JSON.

## 1. Pre-declared extraction sites (unchanged)

| Block (0-indexed) | Equivalent HF index | Layer type | Site |
|---|---|---|---|
| 7 | `hidden_states[8]` | full attention | decoder-block output (residual stream) |
| 15 | `hidden_states[16]` | full attention | same |
| 23 | `hidden_states[24]` | full attention | same (matches the mentor paper's "layer 24") |
| 31 | — | full attention | **decoder-block output before the final RMSNorm.** It is not equal to HF's `hidden_states[32]`, which comes after the norm (max difference 200 on the smoke item). |

The two representations are:
- **last** (primary): the residual stream at the last prompt token, the final `\n` of `<think>\n\n</think>\n\n`. This is pre-decision, since no answer token has been seen. Captures were taken during the ALLOW-continuation pass; with causal attention, positions up to n−1 are independent of the appended label.
- **user_mean** (secondary): the mean over the user-message token span. That span is located by decoding.

## 2. Validation checks

| Check | Result |
|---|---|
| Hooks change nothing | Logits are **bitwise identical** with hooks on and off (smoke test) |
| Hook = HF hidden states | Exact (max difference 0.0) for blocks 7, 15 and 23. Block 31 is pre-norm by design (above). |
| Coverage | 864 rows and 864 representations, all unique. IDs align with the behaviour rows in order. |
| Numerical | All finite. Maximum absolute value 202, against the float16 maximum of 65,504. 864 distinct rows in each of the 8 (block × representation) arrays. |
| Reproducibility | The 48 development items, scored in separate processes, have bitwise-identical label log-probabilities |
| Memory / runtime | Peak MPS driver allocation 25.6 GB. Median forward pass 0.66 s (p95 0.78 s) for 338–460 prompt tokens. One 987 s outlier was an idle-sleep stall. |

Mean L2 norm (SD) of each representation:

| block | last | user_mean |
|---|---|---|
| 7 | 15.8 (0.1) | 17.5 (0.3) |
| 15 | 29.6 (0.3) | 29.9 (0.3) |
| 23 | 71.4 (1.2) | 62.4 (1.0) |
| 31 (pre-norm) | 307.2 (13.5) | 143.2 (4.9) |

The norms grow with depth, as expected for a residual stream. The small SDs show that item-to-item variation is a small fraction of the total norm. Probes therefore standardise features inside each fold.

## 3. Limitations

- Representations are saved as float16, downcast from float32 copies of bf16 activations. bf16 itself has a 7-bit mantissa, so float16 storage adds no meaningful loss.
- Only 4 layers and 2 token positions are captured. Nothing is said about the other 28 layers, the linear-attention layers or the attention heads.
- Decodability from these sites says nothing about causal use.
