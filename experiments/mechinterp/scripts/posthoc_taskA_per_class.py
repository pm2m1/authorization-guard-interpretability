"""POST-HOC EXPLORATORY (D43 addendum): per-class recall and confusion for Task A (oracle-status decoding) on the
primary direct set, for every pre-declared activation spec plus the TF-IDF baseline (same nested LOSO pipeline).
Question: is INCONSISTENT (CONFLICT items, gold ESCALATE, never output by the model) linearly decodable?
Usage: uv run python scripts/posthoc_taskA_per_class.py > results/posthoc_taskA_per_class.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import probes as P


def main():
    rows, reps = P.load()
    R = [r for r in rows if r["wording"] == "direct"]
    groups = np.array([r["base_id"] for r in R])
    y = np.array([r["gold_status"] for r in R], dtype=object)
    D = {"text": [r["user_prompt"] for r in R]}
    for bi, b in enumerate(P.BLOCKS):
        for ri, rp in enumerate(P.REPS):
            D[f"act_{b}_{rp}"] = np.stack([reps[r["item_id"]][bi, ri].astype(np.float32) for r in R])
    out = {"label": "POST-HOC EXPLORATORY Task A per-class (primary direct set)", "classes": P.STATUS, "n_per_class": {
        c: int((y == c).sum()) for c in P.STATUS}}
    specs = {"tfidf": ([], True)} | {f"act_{b}_{rp}": ([f"act_{b}_{rp}"], False) for b in P.BLOCKS for rp in P.REPS}
    for name, (keys, text) in specs.items():
        pred, _ = P.oof(D, y, groups, keys, text, True)
        conf = {g: {p: int(((y == g) & (pred == p)).sum()) for p in P.STATUS} for g in P.STATUS}
        out[name] = {"recall": {c: round(float((pred[y == c] == c).mean()), 3) for c in P.STATUS}, "confusion_gold_rows": conf}
        print(name, out[name]["recall"], file=sys.stderr, flush=True)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
