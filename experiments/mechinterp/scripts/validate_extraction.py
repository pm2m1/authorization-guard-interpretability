"""Extraction validation (report 45): shard/ID integrity, finiteness, float16 range, per-layer norms,
alignment with behaviour records, and the dev-subset forced-choice reproducibility (dev run vs full run)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
BLOCKS = [7, 15, 23, 31]


def main():
    act = HERE / "data/activations"
    rows = [json.loads(line) for line in (HERE / "results/behaviour_full.jsonl").open()]
    meta, rows = rows[0]["_meta"], [r for r in rows if "item_id" in r]
    ids, arrs, shard_hash = [], [], {}
    for f in sorted(act.glob("reps_*.npy")):
        a = np.load(f)
        ids += json.loads((act / (f.stem + ".ids.json")).read_text())
        arrs.append(a)
        shard_hash[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    A = np.concatenate(arrs).astype(np.float32)  # [N, 4, 2, 4096]
    out = {"meta": meta, "n_rows": len(rows), "n_unique_rows": len({r["item_id"] for r in rows}), "n_reps": len(ids),
           "n_unique_reps": len(set(ids)), "ids_match_rows_in_order": ids == [r["item_id"] for r in rows],
           "array_shape": list(A.shape), "all_finite": bool(np.isfinite(A).all()),
           "max_abs_value": float(np.abs(A).max()), "float16_max": 65504.0, "shard_sha256": shard_hash, "per_block": {}}
    for bi, b in enumerate(BLOCKS):
        for ri, rp in enumerate(["last", "user_mean"]):
            X = A[:, bi, ri]
            norms = np.linalg.norm(X, axis=1)
            out["per_block"][f"{b}_{rp}"] = {"norm_mean": round(float(norms.mean()), 1), "norm_sd": round(float(norms.std()), 1),
                                                 "max_abs": round(float(np.abs(X).max()), 1),
                                                 "n_distinct_rows": len(np.unique(X.round(3), axis=0))}
    dev = {json.loads(line)["item_id"]: json.loads(line) for line in (HERE / "results/dev_agreement.jsonl").open() if "item_id" in line}
    full = {r["item_id"]: r for r in rows}
    diffs = [max(abs(dev[i]["label_logprob"][L] - full[i]["label_logprob"][L]) for L in full[i]["label_logprob"]) for i in dev]
    out["dev_vs_full_rerun"] = {"n": len(diffs), "same_decision": sum(dev[i]["decision"] == full[i]["decision"] for i in dev),
                                    "max_abs_logprob_diff": float(max(diffs)), "bitwise_identical": sum(d == 0 for d in diffs)}
    fw = [x for r in rows for x in r["forward_seconds"]]
    out["forward_seconds"] = {"n": len(fw), "median": round(float(np.median(fw)), 3), "p95": round(float(np.quantile(fw, 0.95)), 3),
                                  "max": round(float(max(fw)), 2)}
    out["prompt_tokens"] = {"min": min(r["n_prompt_tokens"] for r in rows), "max": max(r["n_prompt_tokens"] for r in rows)}
    (HERE / "results/extraction_validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "shard_sha256"}, indent=1))


if __name__ == "__main__":
    main()
