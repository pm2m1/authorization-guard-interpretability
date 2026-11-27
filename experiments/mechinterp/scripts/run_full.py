"""Full bounded pilot run (frozen config: configs/pilot_config.json). Two stages:

  stage dev  : pre-declared 48-item scoring-vs-generation agreement check (random.Random(20261126).sample
               over the 864 item_ids in items.jsonl file order). Stops the pipeline if agreement < 90%.
  stage full : forced-choice scoring + residual-stream capture for all 864 items (append-only JSONL;
               representations saved as float16 .npy shards per 96 items).

Usage: HF_HUB_OFFLINE=1 uv run python scripts/run_full.py dev
       HF_HUB_OFFLINE=1 uv run python scripts/run_full.py full
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
from mechinterp import model_io as M

ITEMS = ROOT / "data/synthetic/exploratory_v1/items.jsonl"
SYSTEM = (ROOT / "data/synthetic/exploratory_v1/system_prompt.txt").read_text()
CFG = HERE / "configs/pilot_config.json"
RES = HERE / "results"
ACT = HERE / "data/activations"


def load_items():
    with ITEMS.open() as fh:
        return [json.loads(line) for line in fh]


def dev_ids(items):
    return random.Random(20261126).sample([i["item_id"] for i in items], 48)


def smoke_ids(items):
    return {i["item_id"] for i in items if i["base_id"] == "DA01" and i["family"] in ("existential", "universal")
            and i["condition"] in ("POS", "NEG", "MISS")}


def meta():
    return {"created_utc": datetime.now(UTC).isoformat(), "config_sha256": hashlib.sha256(CFG.read_bytes()).hexdigest(),
                "items_sha256": hashlib.sha256(ITEMS.read_bytes()).hexdigest(), "torch": torch.__version__,
                "model": M.MODEL_ID, "revision": M.REVISION}


def record(it, r, smoke, dev):
    lp = r["label_logprob"]
    vals = torch.tensor([lp[L] for L in M.LABELS])
    probs = torch.softmax(vals, 0)
    return {"item_id": it["item_id"], "base_id": it["base_id"], "domain": it["domain"], "family": it["family"],
                "condition": it["condition"], "wording": it["wording"], "completeness": it["completeness_declared"],
                "pivot_rank": it["pivot_line_rank"], "gold_status": it["oracle_status"], "gold": it["oracle_decision"],
                "n_prompt_tokens": r["n_prompt_tokens"], "label_logprob": lp, "label_prob": {L: float(p) for L, p in zip(M.LABELS, probs)},
                "decision": r["decision"], "margin": r["margin"], "confidence": float(probs.max()),
                "first_token_logprob": r["first_token_logprob"], "first_token_decision": r["first_token_decision"],
                "error": r["decision"] != it["oracle_decision"], "smoke_item": it["item_id"] in smoke, "dev_item": it["item_id"] in dev,
                "forward_seconds": r["forward_seconds"]}


def stage_dev():
    items = load_items()
    ids = set(dev_ids(items))
    smoke = smoke_ids(items)
    tok, model, _ = M.load()
    out = RES / "dev_agreement.jsonl"
    with out.open("w") as fh:
        fh.write(json.dumps({"_meta": meta() | {"stage": "dev", "n": len(ids)}}) + "\n")
        for it in items:
            if it["item_id"] not in ids:
                continue
            r = M.score_item(tok, model, SYSTEM, it["user_prompt"], capture=False)
            gen = M.generate(tok, model, SYSTEM, it["user_prompt"])
            parsed, pstat = M.parse_decision(gen)
            rec = record(it, r, smoke, ids) | {"generated": gen, "generated_decision": parsed, "parse_status": pstat,
                                                   "agree": parsed == r["decision"]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(it["item_id"], r["decision"], parsed, pstat, flush=True)
    rows = [json.loads(line) for line in out.open()][1:]
    agree = sum(r["agree"] for r in rows)
    invalid = sum(r["parse_status"] in ("INVALID", "AMBIGUOUS", "EMPTY") for r in rows)
    summary = {"n": len(rows), "agree": agree, "rate": agree / len(rows), "generation_invalid": invalid,
                   "passes_90pct_rule": agree / len(rows) >= 0.90}
    (RES / "dev_agreement_summary.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY", summary)


def stage_full():
    if not json.loads((RES / "dev_agreement_summary.json").read_text())["passes_90pct_rule"]:
        raise SystemExit("STOP: dev agreement < 90% (pre-declared stopping rule)")
    items = load_items()
    dev = set(dev_ids(items))
    smoke = smoke_ids(items)
    out = RES / "behaviour_full.jsonl"
    done = set()
    if out.exists():
        for line in out.open():
            d = json.loads(line)
            if "item_id" in d:
                done.add(d["item_id"])
    ACT.mkdir(parents=True, exist_ok=True)
    tok, model, _ = M.load()
    shard, shard_ids, shard_no = [], [], len(list(ACT.glob("reps_*.npy")))
    t0 = time.time()
    with out.open("a") as fh:
        if not done:
            fh.write(json.dumps({"_meta": meta() | {"stage": "full", "n": len(items)}}) + "\n")
        for k, it in enumerate(items):
            if it["item_id"] in done:
                continue
            r = M.score_item(tok, model, SYSTEM, it["user_prompt"], capture=True)
            reps = np.stack([np.stack([r["reps"][b]["last"].numpy(), r["reps"][b]["user_mean"].numpy()]) for b in M.BLOCKS])
            shard.append(reps.astype(np.float16))  # shape [4 blocks, 2 reps, 4096]
            shard_ids.append(it["item_id"])
            fh.write(json.dumps(record(it, r, smoke, dev)) + "\n")
            fh.flush()
            if len(shard) == 96 or k == len(items) - 1:
                np.save(ACT / f"reps_{shard_no:02d}.npy", np.stack(shard))
                (ACT / f"reps_{shard_no:02d}.ids.json").write_text(json.dumps(shard_ids))
                shard, shard_ids, shard_no = [], [], shard_no + 1
                print(f"{k + 1}/{len(items)} items, {time.time() - t0:.0f}s, peak driver "
                      f"{torch.mps.driver_allocated_memory() / 1e9:.1f} GB", flush=True)
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    {"dev": stage_dev, "full": stage_full}[sys.argv[1]]()
