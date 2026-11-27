"""Output-label sensitivity diagnostic (reports/48). Pure helpers: item selection, prompt variants, generic
forced-choice scoring over arbitrary candidate strings, and a generic first-line parser.

Conditions (frozen in configs/label_diag_config.json):
  A  original labels ALLOW / DENY / REQUEST_EVIDENCE / ESCALATE (reference; saved results reused after replay check)
  B  ESCALATE -> NEEDS_REVIEW (only the label token changes; rule text and order unchanged)
  C  neutral identifiers, mapping 1: A=ESCALATE, B=ALLOW, C=DENY, D=REQUEST_EVIDENCE
  D  neutral identifiers, mapping 2: A=DENY, B=REQUEST_EVIDENCE, C=ESCALATE, D=ALLOW
In C/D the rules are listed in identifier order (so list position changes with the mapping) and each line states
the operational decision explicitly ("A = ESCALATE: <original rule text>"). Every decision gets a different
identifier and list position in C and D, and none keeps its original position (ALLOW 1, DENY 2, REQ 3, ESC 4).
"""
from __future__ import annotations

import re
import time

import torch

DECISIONS = ("ALLOW", "DENY", "REQUEST_EVIDENCE", "ESCALATE")
RULE_TEXT = {
    "ALLOW": "the evidence establishes that the policy is satisfied.",
    "DENY": "the evidence establishes that the policy is violated.",
    "REQUEST_EVIDENCE": "the evidence is not sufficient to determine whether the policy is satisfied or violated.",
    "ESCALATE": "the evidence contains contradictory statements about a fact the policy depends on.",
}
# candidate string shown/answered for each operational decision, per condition
CANDIDATES = {
    "A": {"ALLOW": "ALLOW", "DENY": "DENY", "REQUEST_EVIDENCE": "REQUEST_EVIDENCE", "ESCALATE": "ESCALATE"},
    "B": {"ALLOW": "ALLOW", "DENY": "DENY", "REQUEST_EVIDENCE": "REQUEST_EVIDENCE", "ESCALATE": "NEEDS_REVIEW"},
    "C": {"ESCALATE": "A", "ALLOW": "B", "DENY": "C", "REQUEST_EVIDENCE": "D"},
    "D": {"DENY": "A", "REQUEST_EVIDENCE": "B", "ESCALATE": "C", "ALLOW": "D"},
}
FAMILIES = ("existential", "universal", "cumulative")
CONTROL_CONDITION = {"TRUE": "POS", "FALSE": "NEG", "UNKNOWN": "MISS"}


def select_items(items: list[dict]) -> list[dict]:
    """All 72 direct CONFLICT items + 72 direct controls: per scenario (sorted base_id, index i) one TRUE (POS),
    one FALSE (NEG) and one UNKNOWN (MISS) control, families rotated so that per scenario the three controls
    cover the three families and across scenarios each (status, family) pair occurs 8 times:
    TRUE -> FAMILIES[i % 3], FALSE -> FAMILIES[(i + 1) % 3], UNKNOWN -> FAMILIES[(i + 2) % 3]."""
    direct = [it for it in items if it["wording"] == "direct"]
    bases = sorted({it["base_id"] for it in direct})
    idx = {(it["base_id"], it["family"], it["condition"]): it for it in direct}
    out = []
    for i, b in enumerate(bases):
        for f in FAMILIES:
            out.append(idx[(b, f, "CONFLICT")] | {"diag_role": "conflict"})
        for k, st in enumerate(("TRUE", "FALSE", "UNKNOWN")):
            f = FAMILIES[(i + k) % 3]
            it = idx[(b, f, CONTROL_CONDITION[st])]
            assert it["oracle_status"] == st, (b, f, st)
            out.append(it | {"diag_role": f"control_{st}"})
    return out


def system_prompt(original: str, cond: str) -> str:
    if cond == "A":
        return original
    if cond == "B":
        assert original.count("ESCALATE") == 2 and "NEEDS_REVIEW" not in original
        return original.replace("ESCALATE", "NEEDS_REVIEW")
    cand = CANDIDATES[cond]
    by_id = sorted(cand.items(), key=lambda kv: kv[1])  # identifier order A, B, C, D
    rules = "\n".join(f"- {ident} = {dec}: {RULE_TEXT[dec]}" for dec, ident in by_id)
    head, rest = original.split("Decision rules:\n", 1)
    _, tail = rest.split("\n\nEvidence semantics:", 1)
    tail = "\n\nEvidence semantics:" + tail
    old_resp = "Respond with exactly one of ALLOW, DENY, REQUEST_EVIDENCE, ESCALATE on the first line"
    assert old_resp in tail
    tail = tail.replace(old_resp, "Respond with exactly one of A, B, C, D on the first line")
    return head + "Decision rules:\n" + rules + tail


@torch.no_grad()
def score(tok, model, render, system: str, user: str, cand: dict, device: str = "mps"):
    """Teacher-forced log-probs of f"{candidate}\\n" for each decision. Returns total, first-token and per-token
    lists plus the token IDs. Prefix is identical across candidates."""
    p_ids = tok(render(tok, system, user), add_special_tokens=False)["input_ids"]
    n = len(p_ids)
    res = {"n_prompt_tokens": n, "tokens": {}, "total": {}, "first": {}, "per_token": {}, "seconds": []}
    for dec in DECISIONS:
        c_ids = tok(f"{cand[dec]}\n", add_special_tokens=False)["input_ids"]
        t0 = time.time()
        logits = model(input_ids=torch.tensor([p_ids + c_ids], device=device)).logits.float()
        if device == "mps":
            torch.mps.synchronize()
        res["seconds"].append(time.time() - t0)
        logp = torch.log_softmax(logits[0], dim=-1)
        lps = [float(logp[n - 1 + j, c_ids[j]]) for j in range(len(c_ids))]
        res["tokens"][dec] = c_ids
        res["per_token"][dec] = lps
        res["total"][dec] = sum(lps)
        res["first"][dec] = lps[0]
    return res


def decide(scores: dict) -> tuple[str, float]:
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return ranked[0][0], ranked[0][1] - ranked[1][1]


def parse_first_line(text: str, cand: dict):
    """Strict: the first non-empty line, stripped of markdown/quotes and an optional 'Decision:' prefix, equals a
    candidate or starts with a candidate followed by a separator. Otherwise INVALID/EMPTY (no lenient search for
    single-letter identifiers, which would match ordinary words)."""
    if not text or not text.strip():
        return None, "EMPTY"
    first = next(line for line in text.splitlines() if line.strip())
    s = re.sub(r"[*`]", "", first).strip().strip("#>\"' ").strip()
    s = re.sub(r"^(decision|answer|verdict)\s*[:\-]\s*", "", s, flags=re.IGNORECASE).strip("*`\"' ").rstrip(".:!").strip()
    inv = {v: k for k, v in cand.items()}
    if s in inv:
        return inv[s], "STRICT"
    for v in sorted(inv, key=len, reverse=True):
        if re.match(rf"^{re.escape(v)}(\s*[:\-–—.,)]|\s+[=(])", s):
            return inv[v], "STRICT"
    return None, "INVALID"
