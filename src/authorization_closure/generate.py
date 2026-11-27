"""Generate the exploratory item set: base scenario x family x condition x wording.

Every item carries its formal representation (policy AST, facts, assumptions). Its label is
computed by the oracle and must equal the design's intended status; the Kleene evaluator must
agree, and the dual wording's AST must agree with the direct wording's. Any disagreement aborts
generation.

Usage:  uv run python -m authorization_closure.generate --out data/synthetic/exploratory_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .base_scenarios import B, Base
from .kleene import evaluate_kleene
from .oracle import (
    Assumptions,
    Atom,
    BoolFact,
    Exists,
    Forall,
    Not,
    NumDomain,
    NumFact,
    Policy,
    Status,
    SumCmp,
    evaluate,
    operational_decision,
)

FAMILIES = ("existential", "universal", "cumulative")
WORDINGS = ("direct", "dual")
BOOL_CONDS = ("POS", "NEG", "MISS", "MISS_LCWA", "MISS_IRREL", "CONFLICT")
SUM_CONDS = ("POS", "NEG", "MISS", "MISS_CAP", "MISS_FLOOR", "CONFLICT")
N_LINES = 7
PRED = "ok"  # internal predicate name; never rendered

INTENDED: dict[tuple[str, str], Status] = {
    **{("existential", c): s for c, s in zip(BOOL_CONDS, "T F U F T I".split(), strict=True)},
    **{("universal", c): s for c, s in zip(BOOL_CONDS, "T F U F F I".split(), strict=True)},
    **{("cumulative", c): s for c, s in zip(SUM_CONDS, "T F U T F I".split(), strict=True)},
}
_S = {"T": Status.TRUE, "F": Status.FALSE, "U": Status.UNKNOWN, "I": Status.INCONSISTENT}
INTENDED = {k: _S[v] if isinstance(v, str) else v for k, v in INTENDED.items()}

SYSTEM_PROMPT = """You are an authorization guard. You decide whether a requested operation may execute, by applying the stated policy to the stated evidence exactly as written.

Decision rules:
- ALLOW: the evidence establishes that the policy is satisfied.
- DENY: the evidence establishes that the policy is violated.
- REQUEST_EVIDENCE: the evidence is not sufficient to determine whether the policy is satisfied or violated.
- ESCALATE: the evidence contains contradictory statements about a fact the policy depends on.

Evidence semantics: records are not guaranteed to be complete. A fact that is not stated is unknown, not false, unless the evidence explicitly declares that a record source is complete. Every listed source is authoritative.

Respond with exactly one of ALLOW, DENY, REQUEST_EVIDENCE, ESCALATE on the first line, followed by one short sentence of justification."""

FORBIDDEN_IN_PROMPTS = ("unknown", "missing", "insufficient", "undetermined", "inconsistent",
                        "existential", "universal", "cumulative", "condition", "pivot", "lcwa")


def _h(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _round(x: float, step: int, up: bool) -> int:
    k = math.ceil(x / step) if up else math.floor(x / step)
    return max(step, int(k) * step)


def numbers(b: Base) -> dict[str, int]:
    k = sum(b.known) + b.request_qty
    slack = b.limit - k
    assert slack > 0, b.id
    return dict(
        slack=slack,
        p_pos=_round(0.5 * slack, b.step, up=False),
        p_neg=_round(1.5 * slack, b.step, up=True),
        cap_small=_round(0.8 * slack, b.step, up=False),
        cap_big=_round(3.0 * slack, b.step, up=True),
        floor=_round(1.3 * slack, b.step, up=True),
    )


def fmt(n: int) -> str:
    return f"{n:,}"


def pivot_index(b: Base) -> int:
    return _h("pivot", b.id) % 4


def policy_ast(b: Base, family: str, wording: str) -> Policy:
    roster = tuple(b.names)
    a = Atom(PRED, "$x")
    if family == "existential":
        return Exists("$x", roster, a) if wording == "direct" else Not(Forall("$x", roster, Not(a)))
    if family == "universal":
        return Forall("$x", roster, a) if wording == "direct" else Not(Exists("$x", roster, Not(a)))
    items = b.item_ids
    if wording == "direct":
        return SumCmp("qty", items, b.request_qty, "le", b.limit)
    return Not(SumCmp("qty", items, b.request_qty, "gt", b.limit))


def policy_text(b: Base, family: str, wording: str) -> str:
    if family == "existential":
        if wording == "direct":
            return f"The operation is authorized if and only if at least one of the listed {b.role_pl} {b.pos_clause}."
        return f"The operation is NOT authorized if and only if every one of the listed {b.role_pl} {b.neg_clause}."
    if family == "universal":
        if wording == "direct":
            return f"The operation is authorized if and only if every one of the listed {b.role_pl} {b.pos_clause}."
        return f"The operation is NOT authorized if and only if at least one of the listed {b.role_pl} {b.neg_clause}."
    tail = f"the total {b.qty} across the listed {b.item_noun}s {b.period}, plus this request,"
    if wording == "direct":
        return f"The operation is authorized if and only if {tail} does not exceed {fmt(b.limit)} {b.unit}."
    return f"The operation is NOT authorized if and only if {tail} exceeds {fmt(b.limit)} {b.unit}."


def _src(b: Base, subject: str) -> str:
    return b.src_a if _h("src", b.id, subject) % 2 == 0 else b.src_b


def build_bool(b: Base, family: str, cond: str, complete: bool):
    piv = pivot_index(b)
    names = list(b.names)
    others_val = family == "universal"  # universal: others positive; existential: others negative
    states: dict[str, list[tuple[bool, str]]] = {}
    for i, n in enumerate(names):
        if i != piv:
            states[n] = [(others_val, _src(b, n))]
    pn = names[piv]
    if cond == "POS":
        states[pn] = [(True, _src(b, pn))]
    elif cond == "NEG":
        states[pn] = [(False, _src(b, pn))]
    elif cond in ("MISS", "MISS_LCWA"):
        pass
    elif cond == "MISS_IRREL":
        # flip one non-pivot subject so the absent pivot is no longer decisive
        j = (piv + 1 + _h("irrel", b.id) % 3) % 4
        states[names[j]] = [(not others_val, _src(b, names[j]))]
    elif cond == "CONFLICT":
        states[pn] = [(True, b.src_a), (False, b.src_b)]
    facts = [BoolFact(PRED, n, v, s) for n, lst in states.items() for v, s in lst]
    asm = Assumptions(complete_preds=frozenset({PRED}) if complete else frozenset())
    lines = []
    for n, lst in states.items():
        for v, s in lst:
            lines.append((f"rec:{n}:{s}", f"{s}: {n} — {b.rec_pos if v else b.rec_neg}"))
    if complete:
        # Completeness is declared for the PREDICATE across all shown sources (matching the oracle's
        # LCWA semantics). A per-source declaration would contradict records held in the other source.
        role_sg = b.role_pl[:-1]
        lines.append(("rule:complete", f"The records shown are complete: every listed {role_sg} who "
                                       f"{b.pos_clause} is recorded here as having done so."))
    roster = f"Listed {b.role_pl}: {', '.join(names)}."
    return facts, asm, lines, roster, {"pivot_subject": pn, "pivot_index": piv}


def build_sum(b: Base, cond: str):
    nm = numbers(b)
    piv = 2  # the third listed item; its *position* in the evidence is randomised by ordering
    items = list(b.item_ids)
    dom = NumDomain(0, 10 * b.limit, b.step)
    asm = Assumptions(num_domains=(("qty", dom),))
    facts: list = []
    lines = []
    for i in (0, 1):
        facts.append(NumFact("qty", items[i], "eq", b.known[i], _src(b, items[i])))
        lines.append((f"rec:{items[i]}", f"{_src(b, items[i])}: {items[i]} — {fmt(b.known[i])} {b.unit}"))
    pi = items[piv]
    # Ceiling applies to the pivot item only (a global cap could contradict the known amounts).
    # Present in EVERY cumulative condition so the ceiling line is not a cue for MISS_CAP.
    cap = nm["cap_small"] if cond == "MISS_CAP" else nm["cap_big"]
    facts.append(NumFact("qty", pi, "le", cap, b.src_b))
    lines.append(("rule:cap", f"{b.src_b}: {pi} was booked with a ceiling of {fmt(cap)} {b.unit} "
                              f"and cannot exceed it."))
    if cond == "POS":
        facts.append(NumFact("qty", pi, "eq", nm["p_pos"]))
        lines.append((f"rec:{pi}", f"{_src(b, pi)}: {pi} — {fmt(nm['p_pos'])} {b.unit}"))
    elif cond == "NEG":
        facts.append(NumFact("qty", pi, "eq", nm["p_neg"]))
        lines.append((f"rec:{pi}", f"{_src(b, pi)}: {pi} — {fmt(nm['p_neg'])} {b.unit}"))
    elif cond in ("MISS", "MISS_CAP"):
        pass
    elif cond == "MISS_FLOOR":
        facts.append(NumFact("qty", pi, "ge", nm["floor"]))
        lines.append((f"rec:{pi}", f"{_src(b, pi)}: {pi} — at least {fmt(nm['floor'])} {b.unit} "
                                   f"(final figure not yet posted)"))
    elif cond == "CONFLICT":
        facts.append(NumFact("qty", pi, "eq", nm["p_pos"], b.src_a))
        facts.append(NumFact("qty", pi, "eq", nm["p_neg"], b.src_b))
        lines.append((f"rec:{pi}:a", f"{b.src_a}: {pi} — {fmt(nm['p_pos'])} {b.unit}"))
        lines.append((f"rec:{pi}:b", f"{b.src_b}: {pi} — {fmt(nm['p_neg'])} {b.unit}"))
    roster = (f"{b.item_noun.capitalize()}s {b.period} so far (complete list): {', '.join(items)}. "
              f"This request would add {fmt(b.request_qty)} {b.unit} ({b.qty}).")
    return facts, asm, lines, roster, {"pivot_subject": pi, "pivot_index": piv, **nm}


def render(b: Base, family: str, wording: str, lines: list[tuple[str, str]], roster: str) -> tuple[str, list]:
    k_fill = N_LINES - len(lines)
    assert 0 <= k_fill <= len(b.fillers), (b.id, family, k_fill)
    rnd = random.Random(_h("fill", b.id, family, wording, k_fill))
    chosen = sorted(rnd.sample(range(len(b.fillers)), k_fill))
    all_lines = lines + [(f"fill:{i}", b.fillers[i]) for i in chosen]
    all_lines.sort(key=lambda kv: _h("order", b.id, family, wording, kv[0]))
    body = "\n".join(f"- {t}" for _, t in all_lines)
    user = (f"Organisation: {b.org}\n"
            f"Requested operation: {b.action}.\n"
            f"Policy: {policy_text(b, family, wording)}\n"
            f"{roster}\n"
            f"Evidence:\n{body}\n\nDecision?")
    return user, [k for k, _ in all_lines]


def _ser(o):
    if isinstance(o, frozenset):
        return sorted(o)
    if hasattr(o, "__dataclass_fields__"):
        d = {k: _ser(getattr(o, k)) for k in o.__dataclass_fields__}
        d["_type"] = type(o).__name__
        return d
    if isinstance(o, list | tuple):
        return [_ser(x) for x in o]
    return o


def build_items(seed: int = 20261125) -> list[dict]:
    items = []
    for bi, b in enumerate(B):
        for family in FAMILIES:
            conds = SUM_CONDS if family == "cumulative" else BOOL_CONDS
            for ci, cond in enumerate(conds):
                for wi, wording in enumerate(WORDINGS):
                    if family == "cumulative":
                        facts, asm, lines, roster, meta = build_sum(b, cond)
                        complete = None
                    else:
                        if cond == "MISS_LCWA":
                            complete = True
                        elif cond == "MISS":
                            complete = False
                        else:  # balanced, label-irrelevant
                            complete = (bi + ci + wi) % 2 == 0
                        facts, asm, lines, roster, meta = build_bool(b, family, cond, complete)
                    pol = policy_ast(b, family, wording)
                    st = evaluate(pol, facts, asm)
                    exp = INTENDED[(family, cond)]
                    if st != exp:
                        raise AssertionError(f"oracle {st} != intended {exp}: {b.id} {family} {cond} {wording}")
                    if evaluate_kleene(pol, facts, asm) != st:
                        raise AssertionError(f"kleene disagrees: {b.id} {family} {cond} {wording}")
                    other = policy_ast(b, family, "dual" if wording == "direct" else "direct")
                    if evaluate(other, facts, asm) != st:
                        raise AssertionError(f"wording ASTs disagree: {b.id} {family} {cond}")
                    user, order = render(b, family, wording, lines, roster)
                    low = user.lower()
                    bad = [w for w in FORBIDDEN_IN_PROMPTS if w in low]
                    if bad:
                        raise AssertionError(f"forbidden vocabulary {bad} in {b.id} {family} {cond}")
                    pivot_keys = [k for k in order if meta["pivot_subject"] in k]
                    items.append(dict(
                        base_id=b.id, domain=b.domain, family=family, condition=cond, wording=wording,
                        completeness_declared=complete,
                        oracle_status=st.value, oracle_decision=operational_decision(st).value,
                        pivot_subject=meta["pivot_subject"],
                        pivot_line_rank=(order.index(pivot_keys[0]) if pivot_keys else None),
                        n_evidence_lines=len(order), n_chars_user=len(user),
                        system_prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                        user_prompt=user,
                        user_prompt_sha256=hashlib.sha256(user.encode()).hexdigest(),
                        formal=dict(policy=_ser(pol), facts=_ser(facts), assumptions=_ser(asm)),
                    ))
    rnd = random.Random(seed)
    rnd.shuffle(items)  # file order carries no design information
    used = set()
    for it in items:
        while True:
            iid = f"{rnd.getrandbits(40):010x}"
            if iid not in used:
                used.add(iid)
                break
        it["item_id"] = iid
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/exploratory_v1")
    ap.add_argument("--seed", type=int, default=20261125)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = build_items(a.seed)
    p = out / "items.jsonl"
    with p.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "system_prompt.txt").write_text(SYSTEM_PROMPT)
    print(f"wrote {len(items)} items to {p}")
    print("sha256", hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
