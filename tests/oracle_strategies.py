"""Hypothesis strategies and shared property checks for the authorization oracle.

The `check_*` functions take an evaluator `ev(policy, facts, asm) -> Status` so the same checks
run against the reference oracle (must pass) and against deliberate mutants (must be caught).
"""

from __future__ import annotations

import random

from hypothesis import strategies as st

from authorization_closure.oracle import (
    And,
    Assumptions,
    Atom,
    BoolFact,
    Exists,
    Forall,
    Not,
    NumDomain,
    NumFact,
    Or,
    Status,
    SumCmp,
)

SUBJ = ("a", "b", "c")
PREDS = ("p", "q")
ITEMS = ("i1", "i2")
DOM = NumDomain(0, 6, 2)  # grid {0,2,4,6}
BASE_ASM = Assumptions(num_domains=(("amt", DOM),))


def _leaf(draw):
    kind = draw(st.sampled_from(["atom", "exists", "forall", "sum"]))
    if kind == "atom":
        a = Atom(draw(st.sampled_from(PREDS)), draw(st.sampled_from(SUBJ)))
        return Not(a) if draw(st.booleans()) else a
    if kind in ("exists", "forall"):
        dom = tuple(draw(st.lists(st.sampled_from(SUBJ), min_size=0, max_size=3, unique=True)))
        body = Atom(draw(st.sampled_from(PREDS)), "$x")
        if draw(st.booleans()):
            body = Not(body)
        return (Exists if kind == "exists" else Forall)("$x", dom, body)
    items = tuple(draw(st.lists(st.sampled_from(ITEMS), min_size=1, max_size=2, unique=True)))
    return SumCmp("amt", items, draw(st.integers(0, 3)), draw(st.sampled_from(["le", "gt"])),
                  draw(st.integers(0, 14)))


@st.composite
def policies(draw):
    left = _leaf(draw)
    if draw(st.booleans()):
        return left
    right = _leaf(draw)
    return (And if draw(st.booleans()) else Or)((left, right))


@st.composite
def facts_lists(draw, max_size=6):
    out = []
    for _ in range(draw(st.integers(0, max_size))):
        if draw(st.booleans()):
            out.append(BoolFact(draw(st.sampled_from(PREDS)), draw(st.sampled_from(SUBJ)), draw(st.booleans())))
        else:
            out.append(NumFact("amt", draw(st.sampled_from(ITEMS)), draw(st.sampled_from(["eq", "ge", "le"])),
                               draw(st.sampled_from([0, 1, 2, 3, 4, 5, 6]))))
    return out


@st.composite
def assumptions(draw, allow_lcwa=True):
    cp = frozenset(draw(st.lists(st.sampled_from(PREDS), unique=True))) if allow_lcwa else frozenset()
    return Assumptions(complete_preds=cp, num_domains=BASE_ASM.num_domains)


# ------------------------------------------------------------------ dual / equivalent forms


def dual(p):
    """A logically equivalent rewrite using quantifier duality / De Morgan / comparison flip."""
    if isinstance(p, Forall):
        return Not(Exists(p.var, p.domain, Not(p.body)))
    if isinstance(p, Exists):
        return Not(Forall(p.var, p.domain, Not(p.body)))
    if isinstance(p, SumCmp):
        return Not(SumCmp(p.attr, p.items, p.extra, "gt" if p.op == "le" else "le", p.limit))
    if isinstance(p, And):
        return Not(Or(tuple(Not(a) for a in p.args)))
    if isinstance(p, Or):
        return Not(And(tuple(Not(a) for a in p.args)))
    if isinstance(p, Not):
        return Not(dual(p.arg))
    return Not(Not(p))


def rename(p, m):
    if isinstance(p, Atom):
        return Atom(p.pred, m.get(p.subject, p.subject))
    if isinstance(p, Not):
        return Not(rename(p.arg, m))
    if isinstance(p, And | Or):
        return type(p)(tuple(rename(a, m) for a in p.args))
    if isinstance(p, Exists | Forall):
        return type(p)(p.var, tuple(m.get(d, d) for d in p.domain), rename(p.body, m))
    return p


def rename_facts(fs, m):
    return [BoolFact(f.pred, m.get(f.subject, f.subject), f.value) if isinstance(f, BoolFact) else f for f in fs]


# ------------------------------------------------------------------ property checks


def check_order_invariance(ev, pol, fs, asm, seed=0) -> bool:
    g = list(fs)
    random.Random(seed).shuffle(g)
    return ev(pol, fs, asm) == ev(pol, g, asm)


def check_rename_invariance(ev, pol, fs, asm) -> bool:
    m = {"a": "zeta", "b": "a", "c": "omega"}  # bijection incl. collision with an old name
    return ev(pol, fs, asm) == ev(rename(pol, m), rename_facts(fs, m), asm)


def check_dual_equivalence(ev, pol, fs, asm) -> bool:
    return ev(pol, fs, asm) == ev(dual(pol), fs, asm)


def check_monotone(ev, pol, fs, extra, asm) -> bool:
    """Adding consistent evidence never turns TRUE/FALSE into UNKNOWN (OWA, or added fact on an
    open predicate). Caller guarantees the precondition on `extra`."""
    before = ev(pol, fs, asm)
    after = ev(pol, fs + [extra], asm)
    if after == Status.INCONSISTENT or before not in (Status.TRUE, Status.FALSE):
        return True
    return after == before


def check_inconsistency_explicit(ev, pol, asm) -> bool:
    fs = [BoolFact("p", "a", True), BoolFact("p", "a", False)]
    return ev(pol, fs, asm) == Status.INCONSISTENT


def random_case(rnd: random.Random):
    """Seeded plain-PRNG analogue of the Hypothesis strategies (for fixed corpora)."""
    def leaf():
        k = rnd.choice(["atom", "exists", "forall", "sum"])
        if k == "atom":
            a = Atom(rnd.choice(PREDS), rnd.choice(SUBJ))
            return Not(a) if rnd.random() < 0.5 else a
        if k in ("exists", "forall"):
            dom = tuple(rnd.sample(SUBJ, rnd.randint(0, 3)))
            body = Atom(rnd.choice(PREDS), "$x")
            body = Not(body) if rnd.random() < 0.5 else body
            return (Exists if k == "exists" else Forall)("$x", dom, body)
        items = tuple(rnd.sample(ITEMS, rnd.randint(1, 2)))
        return SumCmp("amt", items, rnd.randint(0, 3), rnd.choice(["le", "gt"]), rnd.randint(0, 14))

    pol = leaf() if rnd.random() < 0.5 else rnd.choice([And, Or])((leaf(), leaf()))
    fs = []
    for _ in range(rnd.randint(0, 6)):
        if rnd.random() < 0.5:
            fs.append(BoolFact(rnd.choice(PREDS), rnd.choice(SUBJ), rnd.random() < 0.5))
        else:
            fs.append(NumFact("amt", rnd.choice(ITEMS), rnd.choice(["eq", "ge", "le"]), rnd.randint(0, 6)))
    cp = frozenset(p for p in PREDS if rnd.random() < 0.3)
    return pol, fs, Assumptions(complete_preds=cp, num_domains=BASE_ASM.num_domains)
