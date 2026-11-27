"""Runner for the label-sensitivity diagnostic (configs/label_diag_config.json). Stages:
  replay  : condition A on the 8 pre-declared replay items; compare with saved behaviour_full.jsonl (bitwise)
  primary : conditions B, C, D on all 144 items (1,728 scoring passes); append-only JSONL
  freegen : greedy generation on the 12 pre-declared items under A, B, C, D (48 generations)
Usage: HF_HUB_OFFLINE=1 uv run python scripts/label_diag_run.py {replay|primary|freegen}
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
from mechinterp import label_diag as L
from mechinterp import model_io as M

MAN = HERE / "manifests/label_diag"
OUT = HERE / "results/label_diag"
CFG = HERE / "configs/label_diag_config.json"


def items():
    full = {}
    with (ROOT / "data/synthetic/exploratory_v1/items.jsonl").open() as fh:
        for line in fh:
            d = json.loads(line)
            full[d["item_id"]] = d
    with (MAN / "items.jsonl").open() as fh:
        sel = [json.loads(line) for line in fh]
    for s in sel:
        assert hashlib.sha256(full[s["item_id"]]["user_prompt"].encode()).hexdigest() == s["user_prompt_sha256"]
        s["user_prompt"] = full[s["item_id"]]["user_prompt"]
    return sel


def meta(stage):
    return {"_meta": {"stage": stage, "created_utc": datetime.now(UTC).isoformat(), "model": M.MODEL_ID, "revision": M.REVISION,
                      "config_sha256": hashlib.sha256(CFG.read_bytes()).hexdigest()}}


def sysprompt(c):
    return (MAN / f"system_prompt_{c}.txt").read_text()


def main(stage):
    OUT.mkdir(parents=True, exist_ok=True)
    sel = items()
    sub = json.loads((MAN / "subsets.json").read_text())
    orig = (ROOT / "data/synthetic/exploratory_v1/system_prompt.txt").read_text()
    assert sysprompt("A") == orig
    tok, model, _ = M.load()
    if stage == "replay":
        saved = {}
        with (HERE / "results/behaviour_full.jsonl").open() as fh:
            for line in fh:
                d = json.loads(line)
                if "item_id" in d:
                    saved[d["item_id"]] = d
        rows = []
        for s in sel:
            if s["item_id"] not in sub["replay_items"]:
                continue
            r = L.score(tok, model, M.render, orig, s["user_prompt"], L.CANDIDATES["A"])
            rows.append({"item_id": s["item_id"], "new_total": r["total"], "saved_total": saved[s["item_id"]]["label_logprob"],
                         "new_first": r["first"], "saved_first": saved[s["item_id"]]["first_token_logprob"],
                         "bitwise_equal_total": r["total"] == saved[s["item_id"]]["label_logprob"],
                         "bitwise_equal_first": r["first"] == saved[s["item_id"]]["first_token_logprob"]})
        (OUT / "replay_A.json").write_text(json.dumps([meta("replay")] + rows, indent=1))
        print("replay all bitwise equal:", all(x["bitwise_equal_total"] and x["bitwise_equal_first"] for x in rows), len(rows))
    elif stage == "primary":
        path = OUT / "primary_BCD.jsonl"
        done = set()
        if path.exists():
            with path.open() as fh:
                done = {(d["condition_id"], d["item_id"]) for d in map(json.loads, fh) if "item_id" in d}
        with path.open("a") as fh:
            if not done:
                fh.write(json.dumps(meta("primary")) + "\n")
            for c in "BCD":
                sp = sysprompt(c)
                for k, s in enumerate(sel):
                    if (c, s["item_id"]) in done:
                        continue
                    r = L.score(tok, model, M.render, sp, s["user_prompt"], L.CANDIDATES[c])
                    fh.write(json.dumps({"condition_id": c, "item_id": s["item_id"], "base_id": s["base_id"], "diag_role": s["diag_role"],
                                         "gold": s["oracle_decision"]} | {k2: r[k2] for k2 in ("n_prompt_tokens", "tokens", "total", "first", "per_token", "seconds")}) + "\n")
                    fh.flush()
                    if (k + 1) % 48 == 0:
                        print(c, k + 1, flush=True)
    elif stage == "freegen":
        with (OUT / "freegen.jsonl").open("w") as fh:
            fh.write(json.dumps(meta("freegen")) + "\n")
            for c in "ABCD":
                sp = sysprompt(c)
                for s in sel:
                    if s["item_id"] not in sub["freegen_items"]:
                        continue
                    g = M.generate(tok, model, sp, s["user_prompt"])
                    dec, st = L.parse_first_line(g, L.CANDIDATES[c])
                    fh.write(json.dumps({"condition_id": c, "item_id": s["item_id"], "diag_role": s["diag_role"], "gold": s["oracle_decision"],
                                         "generated": g, "parsed_decision": dec, "parse_status": st}) + "\n")
                    fh.flush()
                    print(c, s["item_id"], dec, st, repr(g[:40]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
