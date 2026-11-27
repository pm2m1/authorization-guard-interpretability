"""Analysis for the label-sensitivity diagnostic (frozen plan: configs/label_diag_config.json).
Usage: uv run python scripts/label_diag_analyze.py > results/label_diag/analysis.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from mechinterp.label_diag import DECISIONS

MAN = HERE / "manifests/label_diag"
OUT = HERE / "results/label_diag"
NORMS = ("total", "first", "mean")


def load():
    sel = [json.loads(line) for line in (MAN / "items.jsonl").open()]
    toks = json.loads((MAN / "tokenisation.json").read_text())
    saved = {d["item_id"]: d for d in map(json.loads, (HERE / "results/behaviour_full.jsonl").open()) if "item_id" in d}
    rec = {}
    for s in sel:
        d = saved[s["item_id"]]
        rec[("A", s["item_id"])] = {"total": d["label_logprob"], "first": d["first_token_logprob"],
                                    "n_tok": {k: len(toks["A"][k]["ids"]) for k in DECISIONS}}
    for d in map(json.loads, (OUT / "primary_BCD.jsonl").open()):
        if "item_id" in d:
            rec[(d["condition_id"], d["item_id"])] = {"total": d["total"], "first": d["first"],
                                                      "n_tok": {k: len(v) for k, v in d["tokens"].items()}}
    for r in rec.values():
        r["mean"] = {k: r["total"][k] / r["n_tok"][k] for k in DECISIONS}
    return sel, rec


def softmax(d):
    m = max(d.values())
    z = {k: math.exp(v - m) for k, v in d.items()}
    s = sum(z.values())
    return {k: v / s for k, v in z.items()}


def boot(groups, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(B):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug), replace=True)])
        try:
            vals.append(fn(idx))
        except ValueError:
            continue
    return [round(float(np.quantile(vals, 0.025)), 3), round(float(np.quantile(vals, 0.975)), 3)]


def condition_metrics(sel, rec, c, norm, base=None):
    ids = [s["item_id"] for s in sel]
    gold = np.array([s["oracle_decision"] for s in sel])
    conf = np.array([s["diag_role"] == "conflict" for s in sel])
    groups = np.array([s["base_id"] for s in sel])
    dec = np.array([max(rec[(c, i)][norm], key=rec[(c, i)][norm].get) for i in ids])
    esc = dec == "ESCALATE"
    corr = dec == gold
    out = {"confusion_gold_rows": {g: {d: int(((gold == g) & (dec == d)).sum()) for d in DECISIONS} for g in DECISIONS},
           "conflict_escalations": int(esc[conf].sum()), "conflict_n": int(conf.sum()),
           "escalation_recall": round(float(esc[conf].mean()), 3),
           "escalation_recall_ci": boot(groups, lambda i: esc[i][conf[i]].mean()),
           "control_unnecessary_escalations": int(esc[~conf].sum()), "control_n": int((~conf).sum()),
           "unnecessary_escalation_rate": round(float(esc[~conf].mean()), 3),
           "unnecessary_escalation_ci": boot(groups, lambda i: esc[i][~conf[i]].mean()),
           "unnecessary_escalation_by_status": {st: int(esc[[s["diag_role"] == f"control_{st}" for s in sel]].sum()) for st in ("TRUE", "FALSE", "UNKNOWN")},
           "wrong_allow": {"all": int(((dec == "ALLOW") & (gold != "ALLOW")).sum()), "conflict": int((dec[conf] == "ALLOW").sum()),
                           "controls": int(((dec == "ALLOW") & (gold != "ALLOW") & ~conf).sum())},
           "accuracy": {"all": round(float(corr.mean()), 3), "all_ci": boot(groups, lambda i: corr[i].mean()),
                        "conflict": round(float(corr[conf].mean()), 3), "controls": round(float(corr[~conf].mean()), 3),
                        "controls_ci": boot(groups, lambda i: corr[i][~conf[i]].mean())},
           "decision_distribution": {d: int((dec == d).sum()) for d in DECISIONS}}
    if norm == "total":
        pch = np.array([softmax(rec[(c, i)]["total"])["ESCALATE"] for i in ids])
        pfirst = np.array([math.exp(rec[(c, i)]["first"]["ESCALATE"]) for i in ids])
        out["p_choice_escalate_mean"] = {"conflict": round(float(pch[conf].mean()), 4), "controls": round(float(pch[~conf].mean()), 4),
                                         "conflict_max": round(float(pch[conf].max()), 4)}
        out["p_first_escalate_mean"] = {"conflict": round(float(pfirst[conf].mean()), 4), "controls": round(float(pfirst[~conf].mean()), 4)}
        out["escalate_discrimination_auroc"] = round(float(roc_auc_score(conf, pch)), 3)
        out["escalate_discrimination_auroc_ci"] = boot(groups, lambda i: roc_auc_score(conf[i], pch[i]))
        out["_pch"] = pch
    if base is not None:
        t = {}
        for a, b in zip(base, dec):
            t[f"{a}->{b}"] = t.get(f"{a}->{b}", 0) + 1
        out["transitions_from_A"] = dict(sorted(t.items()))
        ae = base == "ESCALATE"
        out["conflict_gained_escalations"] = int((~ae & esc & conf).sum())
        out["conflict_lost_escalations"] = int((ae & ~esc & conf).sum())
        out["changed_decisions"] = int((base != dec).sum())
        d_rec = lambda i: esc[i][conf[i]].mean() - ae[i][conf[i]].mean()
        out["escalation_recall_change_vs_A"] = round(float(esc[conf].mean() - ae[conf].mean()), 3)
        out["escalation_recall_change_ci"] = boot(groups, d_rec)
        ca = base == gold
        out["accuracy_change_vs_A"] = round(float(corr.mean() - ca.mean()), 3)
        out["accuracy_change_ci"] = boot(groups, lambda i: corr[i].mean() - ca[i].mean())
    return out, dec


def main():
    sel, rec = load()
    res = {"label": "Label-sensitivity diagnostic (exploratory measurement check; not causal)", "n_items": len(sel)}
    for norm in NORMS:
        base = None
        block = {}
        for c in "ABCD":
            m, dec = condition_metrics(sel, rec, c, norm, base)
            if c == "A":
                base = dec
            m.pop("_pch", None)
            block[c] = m
        res[f"norm_{norm}"] = block
    fg = [json.loads(line) for line in (OUT / "freegen.jsonl").open()][1:]
    agree = []
    for g in fg:
        r = rec[(g["condition_id"], g["item_id"])]["total"]
        fc = max(r, key=r.get)
        agree.append({"condition": g["condition_id"], "item_id": g["item_id"], "role": g["diag_role"], "forced": fc,
                      "generated": g["parsed_decision"], "parse_status": g["parse_status"], "agree": fc == g["parsed_decision"],
                      "text": g["generated"][:120]})
    res["freegen_agreement"] = {c: f"{sum(a['agree'] for a in agree if a['condition'] == c)}/{sum(a['condition'] == c for a in agree)}" for c in "ABCD"}
    res["freegen_invalid"] = sum(a["parse_status"] != "STRICT" for a in agree)
    res["freegen_rows"] = agree
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
