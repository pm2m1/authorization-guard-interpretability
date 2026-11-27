"""Section-1 audit for reports/48 (no model inference): per-class precision / recall / F1 and 4x4 confusion for
Task A (oracle-status decoding) on the primary direct set, for the structured baseline, TF-IDF and all 8
pre-declared activation specs (same nested leave-one-scenario-out pipeline as probes.py), plus two
construction-aware ablations of the structured baseline that test INDIRECT identification of status:
  structured_no_condition : family one-hot + completeness flags + pivot rank + prompt length
  structured_meta_only    : completeness flags + pivot rank + prompt length (no family, no condition)
Also tabulates the non-condition structured features by condition.
Usage: uv run python scripts/audit_taskA_classes.py > results/label_diag/audit_taskA.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import probes as P

FEATURE_DEFINITIONS = {
    "family_onehot[3]": "family in (existential, universal, cumulative) — dataset-construction label",
    "condition_onehot[8]": "condition in (POS, NEG, MISS, MISS_LCWA, MISS_IRREL, CONFLICT, MISS_CAP, MISS_FLOOR) — dataset-construction label; CONFLICT <=> INCONSISTENT",
    "completeness_true / completeness_false": "item field completeness_declared == 'True' / == 'False' (whether the evidence declares a source complete)",
    "pivot_rank": "item field pivot_line_rank (position of the decisive evidence line; 0 if None)",
    "n_prompt_tokens/100": "rendered prompt length in tokens / 100",
}


def main():
    rows, reps = P.load()
    R = [r for r in rows if r["wording"] == "direct"]
    groups = np.array([r["base_id"] for r in R])
    y = np.array([r["gold_status"] for r in R], dtype=object)
    S = P.structured(R)
    D = {"text": [r["user_prompt"] for r in R], "struct": S, "struct_nocond": np.hstack([S[:, :3], S[:, 11:]]), "struct_meta": S[:, 11:]}
    for bi, b in enumerate(P.BLOCKS):
        for ri, rp in enumerate(P.REPS):
            D[f"act_{b}_{rp}"] = np.stack([reps[r["item_id"]][bi, ri].astype(np.float32) for r in R])
    out = {"label": "Audit of Task A class-level results (primary direct set, n=432); same pipeline as probes.py",
           "feature_definitions": FEATURE_DEFINITIONS,
           "meta_features_by_condition": {}}
    for cond in sorted({r["condition"] for r in R}):
        rr = [r for r in R if r["condition"] == cond]
        out["meta_features_by_condition"][cond] = {"status": sorted({r["gold_status"] for r in rr}),
                                                   "completeness": dict(Counter(str(r["completeness"]) for r in rr)),
                                                   "pivot_rank": dict(Counter(str(r["pivot_rank"]) for r in rr)),
                                                   "n_prompt_tokens_range": [min(r["n_prompt_tokens"] for r in rr), max(r["n_prompt_tokens"] for r in rr)]}
    specs = {"structured": (["struct"], False), "structured_no_condition": (["struct_nocond"], False),
             "structured_meta_only": (["struct_meta"], False), "tfidf": ([], True)}
    specs |= {f"act_{b}_{rp}": ([f"act_{b}_{rp}"], False) for b in P.BLOCKS for rp in P.REPS}
    for name, (keys, text) in specs.items():
        pred, cs = P.oof(D, y, groups, keys, text, True)
        p, rc, f, sup = precision_recall_fscore_support(y, pred, labels=P.STATUS, zero_division=0)
        out[name] = {"per_class": {c: {"precision": round(float(p[i]), 3), "recall": round(float(rc[i]), 3), "f1": round(float(f[i]), 3),
                                       "support": int(sup[i])} for i, c in enumerate(P.STATUS)},
                     "accuracy": round(float((pred == y).mean()), 3),
                     "confusion_gold_rows": {g: {q: int(((y == g) & (pred == q)).sum()) for q in P.STATUS} for g in P.STATUS},
                     "chosen_C": sorted(set(cs))}
        print(name, out[name]["per_class"]["INCONSISTENT"], file=sys.stderr, flush=True)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
