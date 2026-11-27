"""POST-HOC EXPLORATORY (not pre-declared; added after the Task B point estimates were seen, D43).
Question: within a (family, condition) cell the structured features are constant, so any error-ranking ability
there must come from something else. For cells with >=5 errors and >=5 non-errors, AUROC of pooled out-of-fold
scores (from probes.py --save-oof) for margin, text+margin and each activation probe. Few cells qualify; the
results are descriptive only, with no CI-based claims.
Usage: uv run python scripts/posthoc_within_cell.py results/oof_scores.json > results/posthoc_within_cell.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]


def main():
    oof = json.loads(Path(sys.argv[1]).read_text())
    rows = {json.loads(line)["item_id"]: json.loads(line) for line in (HERE / "results/behaviour_full.jsonl").open() if '"item_id"' in line}
    out = {"label": "POST-HOC EXPLORATORY within-cell error ranking (D43); descriptive only"}
    for set_name, d in oof.items():
        ids = d["item_id"]
        y = np.array(d["error"])
        cell = np.array([f"{rows[i]['family']}|{rows[i]['condition']}" for i in ids])
        res = {}
        for c in sorted(set(cell)):
            m = cell == c
            ne, nn = int(y[m].sum()), int((1 - y[m]).sum())
            if ne >= 5 and nn >= 5:
                res[c] = {"errors": ne, "non_errors": nn,
                          "auroc": {k: round(float(roc_auc_score(y[m], np.array(v)[m])), 3) for k, v in d["scores"].items()
                                    if k in ("margin(-margin)", "text+margin", "tfidf") or k.startswith("act_")}}
        out[set_name] = res
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
