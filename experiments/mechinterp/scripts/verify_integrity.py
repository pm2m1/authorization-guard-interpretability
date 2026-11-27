"""Inference-free integrity check for the preserved mechinterp raw results (housekeeping, D47).

Uses only the Python standard library and the files in this repository; it loads no model and runs no experiment.
Checks:
  1. SHA-256 of every file listed in manifests/raw_results.sha256 (paths relative to experiments/mechinterp).
  2. results/behaviour_full.jsonl (original condition-A raw scores, 864 items): unique item_ids; the stored decision
     equals the argmax of the stored total log-probs, and first_token_decision the argmax of first-token log-probs.
  3. manifests/label_diag/items.jsonl (144 diagnostic items): every item_id is present in the raw file with identical
     metadata and gold labels, and user_prompt_sha256 equals the SHA-256 of the source item's user_prompt.
  4. Independent recomputation from the raw log-probs of the condition-A headline counts:
     0/72 correct escalations and 15/72 wrong ALLOW on CONFLICT items; 0/72 unnecessary escalations on controls;
     0/864 ESCALATE decisions overall.
  5. The 8-item replay (results/label_diag/replay_A.json) is bitwise equal to the raw file, and the condition-A block
     of results/label_diag/analysis.json agrees with the recomputation.
Exit status 0 if all checks pass, 1 otherwise. Usage (from experiments/mechinterp): python3 scripts/verify_integrity.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LABELS = ("ALLOW", "DENY", "REQUEST_EVIDENCE", "ESCALATE")
EXPECTED = {"conflict_n": 72, "control_n": 72, "conflict_escalations": 0, "conflict_wrong_allow": 15,
            "control_unnecessary_escalations": 0, "escalate_decisions_all": 0}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argmax(scores: dict) -> str:
    return max(LABELS, key=lambda k: scores[k])


def main() -> int:
    failures: list[str] = []
    out: dict = {}

    def check(ok: bool, msg: str) -> None:
        if not ok:
            failures.append(msg)

    # 1. checksums
    n_sum = 0
    for line in (HERE / "manifests/raw_results.sha256").read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        path = (HERE / rel).resolve()
        check(path.exists(), f"missing: {rel}")
        if path.exists():
            check(sha256(path) == digest, f"sha256 mismatch: {rel}")
        n_sum += 1
    out["checksummed_files"] = n_sum

    # 2. raw condition-A scores
    rows = [json.loads(line) for line in (HERE / "results/behaviour_full.jsonl").open()]
    meta, recs = rows[0]["_meta"], rows[1:]
    raw = {r["item_id"]: r for r in recs}
    check(meta["n"] == len(recs) == len(raw) == 864, "raw file: expected 864 unique items")
    check(all(argmax(r["label_logprob"]) == r["decision"] for r in recs), "raw file: decision != argmax(total)")
    check(all(argmax(r["first_token_logprob"]) == r["first_token_decision"] for r in recs),
          "raw file: first_token_decision != argmax(first)")
    out["raw_items"] = len(recs)

    # 3. diagnostic items <-> raw scores <-> source prompts
    items = [json.loads(line) for line in (HERE / "manifests/label_diag/items.jsonl").open()]
    src = {d["item_id"]: d for d in map(json.loads, (HERE / "../../data/synthetic/exploratory_v1/items.jsonl").open())}
    check(len(items) == len({i["item_id"] for i in items}) == 144, "diag items: expected 144 unique ids")
    matched = 0
    for it in items:
        r = raw.get(it["item_id"])
        s = src.get(it["item_id"])
        ok = (r is not None and s is not None
              and all(r[k] == it[k] for k in ("base_id", "family", "condition", "wording"))
              and r["gold"] == it["oracle_decision"] and r["gold_status"] == it["oracle_status"]
              and hashlib.sha256(s["user_prompt"].encode()).hexdigest() == it["user_prompt_sha256"])
        check(ok, f"diag item not matched: {it['item_id']}")
        matched += ok
    out["diag_items_matched"] = matched

    # 4. recompute condition-A counts from raw log-probs
    conflict = [raw[i["item_id"]] for i in items if i["diag_role"] == "conflict"]
    control = [raw[i["item_id"]] for i in items if i["diag_role"] != "conflict"]
    got = {"conflict_n": len(conflict), "control_n": len(control),
           "conflict_escalations": sum(argmax(r["label_logprob"]) == "ESCALATE" for r in conflict),
           "conflict_wrong_allow": sum(argmax(r["label_logprob"]) == "ALLOW" for r in conflict),
           "control_unnecessary_escalations": sum(argmax(r["label_logprob"]) == "ESCALATE" for r in control),
           "escalate_decisions_all": sum(argmax(r["label_logprob"]) == "ESCALATE" for r in recs)}
    for k, v in EXPECTED.items():
        check(got[k] == v, f"recomputed {k} = {got[k]}, reported {v}")
    direct_conflict = {r["item_id"] for r in recs if r["condition"] == "CONFLICT" and r["wording"] == "direct"}
    check(direct_conflict == {r["item_id"] for r in conflict}, "diag conflict set != all 72 direct CONFLICT items")

    def p_esc(r):
        z = [math.exp(v) for v in r["label_logprob"].values()]
        return math.exp(r["label_logprob"]["ESCALATE"]) / sum(z)

    got["max_p_choice_escalate_all"] = round(max(map(p_esc, recs)), 4)
    got["conflict_decisions"] = dict(Counter(argmax(r["label_logprob"]) for r in conflict))
    got["first_token_conflict_escalations"] = sum(argmax(r["first_token_logprob"]) == "ESCALATE" for r in conflict)
    got["first_token_conflict_wrong_allow"] = sum(argmax(r["first_token_logprob"]) == "ALLOW" for r in conflict)
    out["recomputed_condition_A"] = got

    # 5. replay and saved analysis consistency
    replay = json.load((HERE / "results/label_diag/replay_A.json").open())[1:]
    check(len(replay) == 8, "replay: expected 8 items")
    check(all(x["bitwise_equal_total"] and x["bitwise_equal_first"] for x in replay), "replay: not bitwise equal")
    check(all(x["saved_total"] == raw[x["item_id"]]["label_logprob"]
              and x["saved_first"] == raw[x["item_id"]]["first_token_logprob"] for x in replay),
          "replay: saved scores differ from raw file")
    a = json.load((HERE / "results/label_diag/analysis.json").open())["norm_total"]["A"]
    check(a["conflict_escalations"] == got["conflict_escalations"] and a["conflict_n"] == 72
          and a["control_unnecessary_escalations"] == got["control_unnecessary_escalations"]
          and a["wrong_allow"]["conflict"] == got["conflict_wrong_allow"], "analysis.json condition A disagrees")

    out["failures"] = failures
    out["status"] = "PASS" if not failures else "FAIL"
    print(json.dumps(out, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
