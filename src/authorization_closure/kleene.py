"""Independent second evaluator: strong-Kleene three-valued logic + interval arithmetic.

Written separately from oracle.py (no shared evaluation code; only the AST/fact dataclasses
are imported). It is EXACT only for *linear* policies, where every ground atom occurs at most
once after quantifier expansion. All policy shapes used in the experiment are linear.
For non-linear formulas (e.g. p OR NOT p) Kleene logic is sound but incomplete: it may return
UNKNOWN where enumeration returns TRUE. tests/unit/test_oracle_crosscheck.py documents this.
"""

from __future__ import annotations

from collections import Counter

from .oracle import (
    And,
    Assumptions,
    Atom,
    BoolFact,
    Exists,
    Fact,
    Forall,
    Not,
    NumFact,
    Or,
    Policy,
    Status,
    SumCmp,
    substitute,
)

T, F, U = "T", "F", "U"


class _Inconsistent(Exception):
    pass


def _bool_value(pred: str, subj: str, facts: list[Fact], asm: Assumptions) -> str:
    vals = {f.value for f in facts if isinstance(f, BoolFact) and f.pred == pred and f.subject == subj}
    if len(vals) == 2:
        raise _Inconsistent
    if vals:
        return T if True in vals else F
    return F if pred in asm.complete_preds else U


def _interval(attr: str, item: str, facts: list[Fact], asm: Assumptions) -> tuple[int, int]:
    d = asm.domain(attr)
    lo, hi = d.lo, d.lo + ((d.hi - d.lo) // d.step) * d.step  # largest grid value <= hi
    for f in facts:
        if not (isinstance(f, NumFact) and f.attr == attr and f.item == item):
            continue
        if f.op == "eq":
            if (f.value - d.lo) % d.step != 0:
                raise _Inconsistent
            lo, hi = max(lo, f.value), min(hi, f.value)
        elif f.op == "ge":
            # smallest grid point >= value
            k = -(-(f.value - d.lo) // d.step)
            lo = max(lo, d.lo + max(k, 0) * d.step)
        elif f.op == "le":
            k = (f.value - d.lo) // d.step
            hi = min(hi, d.lo + k * d.step)
    if lo > hi:
        raise _Inconsistent
    return lo, hi


def _not(v: str) -> str:
    return {T: F, F: T, U: U}[v]


def _eval(p: Policy, facts: list[Fact], asm: Assumptions) -> str:
    if isinstance(p, Atom):
        return _bool_value(p.pred, p.subject, facts, asm)
    if isinstance(p, Not):
        return _not(_eval(p.arg, facts, asm))
    if isinstance(p, And | Forall):
        parts = (
            [_eval(a, facts, asm) for a in p.args]
            if isinstance(p, And)
            else [_eval(substitute(p.body, p.var, d), facts, asm) for d in p.domain]
        )
        if F in parts:
            return F
        return U if U in parts else T
    if isinstance(p, Or | Exists):
        parts = (
            [_eval(a, facts, asm) for a in p.args]
            if isinstance(p, Or)
            else [_eval(substitute(p.body, p.var, d), facts, asm) for d in p.domain]
        )
        if T in parts:
            return T
        return U if U in parts else F
    if isinstance(p, SumCmp):
        ivs = [_interval(p.attr, i, facts, asm) for i in p.items]
        mn = sum(a for a, _ in ivs) + p.extra
        mx = sum(b for _, b in ivs) + p.extra
        if p.op == "le":
            return T if mx <= p.limit else (F if mn > p.limit else U)
        return T if mn > p.limit else (F if mx <= p.limit else U)
    raise TypeError(p)


def _check_evidence_consistency(facts: list[Fact], asm: Assumptions) -> None:
    # Any contradiction anywhere in the evidence empties W(E), even on atoms the policy ignores.
    for f in facts:
        if isinstance(f, BoolFact):
            _bool_value(f.pred, f.subject, facts, asm)
        else:
            _interval(f.attr, f.item, facts, asm)


def evaluate_kleene(policy: Policy, facts: list[Fact], asm: Assumptions) -> Status:
    try:
        _check_evidence_consistency(facts, asm)
        v = _eval(policy, facts, asm)
    except _Inconsistent:
        return Status.INCONSISTENT
    return {T: Status.TRUE, F: Status.FALSE, U: Status.UNKNOWN}[v]


def is_linear(policy: Policy) -> bool:
    """True if every ground atom occurs at most once after quantifier expansion."""
    from .oracle import expand

    c: Counter = Counter()

    def walk(q):
        if isinstance(q, Atom):
            c[("b", q.pred, q.subject)] += 1
        elif isinstance(q, Not):
            walk(q.arg)
        elif isinstance(q, And | Or):
            for a in q.args:
                walk(a)
        elif isinstance(q, SumCmp):
            for i in q.items:
                c[("n", q.attr, i)] += 1

    walk(expand(policy))
    return all(v == 1 for v in c.values())
