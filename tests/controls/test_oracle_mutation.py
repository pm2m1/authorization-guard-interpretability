"""MUTATION ANALYSIS — deliberately defective oracles (clearly labelled, never used elsewhere).

Each mutant encodes a plausible real defect. The property checks / control cases used for the
reference oracle must detect (kill) every mutant on a fixed, seeded corpus.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from authorization_closure import oracle as O
from authorization_closure.oracle import (
    Assumptions,
    Atom,
    BoolFact,
    Decision,
    Exists,
    Forall,
    NumDomain,
    NumFact,
    Status,
    SumCmp,
    evaluate,
)
from oracle_strategies import (
    BASE_ASM,
    check_dual_equivalence,
    check_inconsistency_explicit,
    check_order_invariance,
    random_case,
)

pytestmark = pytest.mark.mutation

# ------------------------------------------------------------------ MUTANTS (defective!)


def MUT_closed_world(pol, fs, asm):
    """Treats every unmentioned boolean fact as FALSE (CWA leak)."""
    preds = {f.pred for f in fs if isinstance(f, BoolFact)} | {"p", "q", "approved"}
    return evaluate(pol, fs, replace(asm, complete_preds=frozenset(preds)))


def MUT_ignore_lcwa(pol, fs, asm):
    """Ignores declared completeness."""
    return evaluate(pol, fs, replace(asm, complete_preds=frozenset()))


def MUT_vacuous_inconsistency(pol, fs, asm):
    """Checks satisfaction before emptiness: empty W(E) yields TRUE (vacuous forall)."""
    ground = O.expand(pol)
    worlds = list(O.possible_worlds(pol, fs, asm))
    if all(O.holds(ground, wb, wn) for wb, wn in worlds):
        return Status.TRUE
    if not any(O.holds(ground, wb, wn) for wb, wn in worlds):
        return Status.FALSE
    return Status.UNKNOWN


def MUT_missing_as_zero(pol, fs, asm):
    """Unstated numeric amounts are assumed to be zero."""
    _, natoms = O.atoms_of(pol)
    stated = {(f.attr, f.item) for f in fs if isinstance(f, NumFact)}
    extra = [NumFact(a, i, "eq", 0) for (a, i) in natoms if (a, i) not in stated]
    return evaluate(pol, fs + extra, asm)


def MUT_strict_limit(pol, fs, asm):
    """Off-by-one: uses total < limit instead of total <= limit."""
    def tweak(p):
        if isinstance(p, SumCmp):
            return SumCmp(p.attr, p.items, p.extra, p.op, p.limit - 1 if p.op == "le" else p.limit)
        return p
    return evaluate(tweak(pol), fs, asm)


def MUT_order_dependent(pol, fs, asm):
    """Last-writer-wins on conflicting boolean facts (hides inconsistency, order dependent)."""
    last = {}
    for f in fs:
        if isinstance(f, BoolFact):
            last[(f.pred, f.subject)] = f
    clean = [f for f in fs if not isinstance(f, BoolFact)] + list(last.values())
    return evaluate(pol, clean, asm)


MUT_OPERATIONAL = {Status.TRUE: Decision.ALLOW, Status.FALSE: Decision.DENY,
                   Status.UNKNOWN: Decision.ALLOW, Status.INCONSISTENT: Decision.ESCALATE}

# ------------------------------------------------------------------ fixed detection corpus

R = ("r1", "r2", "r3")
AMT = Assumptions(num_domains=(("amt", NumDomain(0, 10_000, 100)),))
SUMPOL = SumCmp("amt", ("t1", "t2", "t3"), 2_000, "le", 10_000)
ANCHORS = [  # (policy, facts, asm, expected) — expected values derived by hand
    (Exists("$x", R, Atom("p", "$x")), [BoolFact("p", "r1", False), BoolFact("p", "r2", False)], BASE_ASM,
     Status.UNKNOWN),
    (Forall("$x", R, Atom("p", "$x")), [BoolFact("p", "r1", True), BoolFact("p", "r2", True)],
     Assumptions(complete_preds=frozenset({"p"})), Status.FALSE),
    (SUMPOL, [NumFact("amt", "t1", "eq", 3000), NumFact("amt", "t2", "eq", 2000)], AMT, Status.UNKNOWN),
    (SUMPOL, [NumFact("amt", "t1", "eq", 3000), NumFact("amt", "t2", "eq", 2000),
              NumFact("amt", "t3", "eq", 3000)], AMT, Status.TRUE),
]


def _random_cases(n=300, seed=7):
    """Deterministic corpus from a seeded PRNG (independent of Hypothesis' database)."""
    rnd = random.Random(seed)
    return [random_case(rnd) for _ in range(n)]


def killed(ev) -> bool:
    for pol, fs, asm, exp in ANCHORS:
        if ev(pol, fs, asm) != exp:
            return True
    for pol in [Forall("$x", (), Atom("p", "$x")), SUMPOL]:
        if not check_inconsistency_explicit(ev, pol, BASE_ASM):
            return True
    for pol, fs, asm in CASES:
        if not check_order_invariance(ev, pol, fs, asm, seed=1):
            return True
        if not check_dual_equivalence(ev, pol, fs, asm):
            return True
        if ev(pol, fs, asm) != evaluate(pol, fs, asm):  # differential vs reference
            return True
    return False


CASES: list = []


@pytest.fixture(scope="module", autouse=True)
def _corpus():
    CASES.extend(_random_cases())
    yield


def test_reference_oracle_survives_corpus():
    assert not killed(evaluate)


@pytest.mark.parametrize("mutant", [MUT_closed_world, MUT_ignore_lcwa, MUT_vacuous_inconsistency,
                                    MUT_missing_as_zero, MUT_strict_limit, MUT_order_dependent])
def test_mutant_is_killed(mutant):
    assert killed(mutant), f"test corpus failed to detect {mutant.__name__}"


def test_operational_mutant_is_killed():
    # property 9 check, applied to the mutant mapping
    assert any((MUT_OPERATIONAL[s] == Decision.ALLOW) != (s == Status.TRUE) for s in Status)


def test_mutants_killed_by_anchors_alone():
    """Stronger requirement: the hand-derived anchors (no differential vs reference) must kill the
    semantic mutants, so detection does not rest only on comparing to the oracle itself."""
    def anchors_kill(ev):
        if any(ev(p, f, a) != e for p, f, a, e in ANCHORS):
            return True
        return not all(check_inconsistency_explicit(ev, p, BASE_ASM)
                       for p in [Forall("$x", (), Atom("p", "$x")), SUMPOL])
    for m in [MUT_closed_world, MUT_ignore_lcwa, MUT_vacuous_inconsistency, MUT_missing_as_zero,
              MUT_strict_limit, MUT_order_dependent]:
        assert anchors_kill(m), m.__name__
