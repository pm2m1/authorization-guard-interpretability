"""Ground-truth tests for the possible-world oracle (properties 1-10 of the protocol)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from authorization_closure.kleene import evaluate_kleene, is_linear
from authorization_closure.oracle import (
    OPERATIONAL_POLICY,
    And,
    Assumptions,
    Atom,
    BoolFact,
    Decision,
    Exists,
    Forall,
    Not,
    NumDomain,
    NumFact,
    Or,
    Status,
    SumCmp,
    evaluate,
    operational_decision,
)
from oracle_strategies import (
    BASE_ASM,
    ITEMS,
    PREDS,
    SUBJ,
    assumptions,
    check_dual_equivalence,
    check_inconsistency_explicit,
    check_monotone,
    check_order_invariance,
    check_rename_invariance,
    facts_lists,
    policies,
)

S = settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])

# --- (1)/(2) monotonicity under fixed assumptions -----------------------------------------


@S
@given(policies(), facts_lists(), facts_lists(max_size=1), assumptions(allow_lcwa=False))
def test_p1_p2_adding_consistent_evidence_preserves_determinate_status_owa(pol, fs, extra, asm):
    assume(extra)
    assert check_monotone(evaluate, pol, fs, extra[0], asm)


@S
@given(policies(), facts_lists(), facts_lists(max_size=1), assumptions())
def test_p1_p2_monotone_under_lcwa_when_added_fact_is_on_open_predicate(pol, fs, extra, asm):
    assume(extra)
    f = extra[0]
    assume(not (isinstance(f, BoolFact) and f.pred in asm.complete_preds))
    assert check_monotone(evaluate, pol, fs, f, asm)


def test_lcwa_is_nonmonotone_documented():
    """Stated assumption: with a locally-closed predicate, adding a positive fact CAN flip
    FALSE->TRUE (the closure itself changes). This is why P1/P2 are restricted above."""
    asm = Assumptions(complete_preds=frozenset({"p"}))
    pol = Atom("p", "a")
    assert evaluate(pol, [], asm) == Status.FALSE
    assert evaluate(pol, [BoolFact("p", "a", True)], asm) == Status.TRUE


# --- (3) order, (4) renaming, (5) equivalent representations --------------------------------


@S
@given(policies(), facts_lists(), assumptions(), st.integers(0, 10_000))
def test_p3_evidence_order_invariance(pol, fs, asm, seed):
    assert check_order_invariance(evaluate, pol, fs, asm, seed)


@S
@given(policies(), facts_lists(), assumptions())
def test_p4_entity_renaming_invariance(pol, fs, asm):
    assert check_rename_invariance(evaluate, pol, fs, asm)


@S
@given(policies(), facts_lists(), assumptions())
def test_p5_logically_equivalent_policies_same_verdict(pol, fs, asm):
    assert check_dual_equivalence(evaluate, pol, fs, asm)


# --- (6) unknown is not false under OWA ----------------------------------------------------


@pytest.mark.parametrize("pol", [Atom("p", "a"), Not(Atom("p", "a")),
                                 Exists("$x", ("a", "b"), Atom("p", "$x")),
                                 Forall("$x", ("a", "b"), Atom("p", "$x"))])
def test_p6_unmentioned_atom_is_unknown_not_false(pol):
    assert evaluate(pol, [], BASE_ASM) == Status.UNKNOWN


def test_p6_existential_with_all_but_one_negative_is_unknown():
    fs = [BoolFact("p", "a", False), BoolFact("p", "b", False)]
    assert evaluate(Exists("$x", ("a", "b", "c"), Atom("p", "$x")), fs, BASE_ASM) == Status.UNKNOWN


def test_p6_universal_with_all_but_one_positive_is_unknown():
    fs = [BoolFact("p", "a", True), BoolFact("p", "b", True)]
    assert evaluate(Forall("$x", ("a", "b", "c"), Atom("p", "$x")), fs, BASE_ASM) == Status.UNKNOWN


# --- (7) inconsistency explicit, never vacuous TRUE/FALSE ------------------------------------


@pytest.mark.parametrize("pol", [Forall("$x", (), Atom("p", "$x")), Exists("$x", (), Atom("p", "$x")),
                                 Atom("q", "b"), SumCmp("amt", ("i1",), 0, "le", 6)])
def test_p7_inconsistent_evidence_is_explicit_even_for_vacuous_policies(pol):
    assert check_inconsistency_explicit(evaluate, pol, BASE_ASM)


def test_p7_numeric_conflict_is_inconsistent():
    fs = [NumFact("amt", "i1", "eq", 2), NumFact("amt", "i1", "eq", 4)]
    assert evaluate(SumCmp("amt", ("i1",), 0, "le", 100), fs, BASE_ASM) == Status.INCONSISTENT
    fs2 = [NumFact("amt", "i1", "ge", 5), NumFact("amt", "i1", "le", 3)]
    assert evaluate(SumCmp("amt", ("i1",), 0, "le", 100), fs2, BASE_ASM) == Status.INCONSISTENT


def test_p7_vacuous_policies_with_consistent_evidence():
    assert evaluate(Forall("$x", (), Atom("p", "$x")), [], BASE_ASM) == Status.TRUE
    assert evaluate(Exists("$x", (), Atom("p", "$x")), [], BASE_ASM) == Status.FALSE


# --- (8) numerical bounds ------------------------------------------------------------------

AMT = Assumptions(num_domains=(("amt", NumDomain(0, 10_000, 100)),))
POL = SumCmp("amt", ("t1", "t2", "t3"), 2_000, "le", 10_000)  # prior 3 + request 2000 <= 10000


def _known(t1, t2):
    return [NumFact("amt", "t1", "eq", t1), NumFact("amt", "t2", "eq", t2)]


def test_p8_missing_amount_unbounded_is_unknown_and_missing_as_zero_would_allow():
    fs = _known(3_000, 2_000)  # known 5000 + req 2000 = 7000 <= 10000; t3 unknown in [0, 10000]
    assert evaluate(POL, fs, AMT) == Status.UNKNOWN
    assert evaluate(POL, fs + [NumFact("amt", "t3", "eq", 0)], AMT) == Status.TRUE  # the zero trap


def test_p8_upper_bound_sufficient_for_satisfaction():
    fs = _known(3_000, 2_000) + [NumFact("amt", "t3", "le", 3_000)]
    assert evaluate(POL, fs, AMT) == Status.TRUE
    fs_edge = _known(3_000, 2_000) + [NumFact("amt", "t3", "le", 3_100)]
    assert evaluate(POL, fs_edge, AMT) == Status.UNKNOWN


def test_p8_lower_bound_sufficient_for_violation():
    fs = _known(3_000, 2_000) + [NumFact("amt", "t3", "ge", 3_100)]
    assert evaluate(POL, fs, AMT) == Status.FALSE
    fs_edge = _known(3_000, 2_000) + [NumFact("amt", "t3", "ge", 3_000)]
    assert evaluate(POL, fs_edge, AMT) == Status.UNKNOWN


def test_p8_known_part_alone_exceeds_limit_is_false_despite_missing():
    fs = _known(5_000, 3_500)  # 8500 + 2000 > 10000 regardless of t3 >= 0
    assert evaluate(POL, fs, AMT) == Status.FALSE


def test_p8_boundary_equality_satisfies_le():
    fs = _known(3_000, 2_000) + [NumFact("amt", "t3", "eq", 3_000)]  # exactly 10000
    assert evaluate(POL, fs, AMT) == Status.TRUE


def test_p8_off_grid_and_invalid_bounds():
    assert evaluate(POL, [NumFact("amt", "t1", "eq", 150)], AMT) == Status.INCONSISTENT
    assert evaluate(POL, [NumFact("amt", "t1", "le", -1)], AMT) == Status.INCONSISTENT


# --- (9) operational policy ----------------------------------------------------------------


def test_p9_only_true_authorizes():
    for s in Status:
        assert (operational_decision(s) == Decision.ALLOW) == (s == Status.TRUE)
    assert OPERATIONAL_POLICY[Status.UNKNOWN] != Decision.ALLOW
    assert OPERATIONAL_POLICY[Status.INCONSISTENT] != Decision.ALLOW


# --- (10) synthetic positive / negative controls -------------------------------------------

ROSTER = ("r1", "r2", "r3", "r4")
EX = Exists("$x", ROSTER, Atom("approved", "$x"))
ALL = Forall("$x", ROSTER, Atom("approved", "$x"))
OWA = Assumptions()
LCWA = Assumptions(complete_preds=frozenset({"approved"}))


def _f(**states):
    return [BoolFact("approved", s, v) for s, v in states.items()]


@pytest.mark.parametrize("pol,facts,asm,expected", [
    (EX, _f(r1=False, r2=False, r3=False, r4=True), OWA, Status.TRUE),
    (EX, _f(r1=False, r2=False, r3=False, r4=False), OWA, Status.FALSE),
    (EX, _f(r1=False, r2=False, r3=False), OWA, Status.UNKNOWN),
    (EX, _f(r1=False, r2=False, r3=False), LCWA, Status.FALSE),
    (EX, _f(r1=True, r2=False, r3=False), OWA, Status.TRUE),  # missing but irrelevant
    (EX, _f(r1=False, r2=False, r3=False) + _f(r4=True) + _f(r4=False), OWA, Status.INCONSISTENT),
    (ALL, _f(r1=True, r2=True, r3=True, r4=True), OWA, Status.TRUE),
    (ALL, _f(r1=True, r2=True, r3=True, r4=False), OWA, Status.FALSE),
    (ALL, _f(r1=True, r2=True, r3=True), OWA, Status.UNKNOWN),
    (ALL, _f(r1=True, r2=True, r3=True), LCWA, Status.FALSE),
    (ALL, _f(r1=True, r2=False, r3=True), OWA, Status.FALSE),  # missing but irrelevant
])
def test_p10_design_controls(pol, facts, asm, expected):
    assert evaluate(pol, facts, asm) == expected
    assert evaluate_kleene(pol, facts, asm) == expected


# --- independent cross-check --------------------------------------------------------------


@settings(max_examples=1500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(policies(), facts_lists(), assumptions())
def test_crosscheck_enumeration_vs_kleene_on_linear_policies(pol, fs, asm):
    assume(is_linear(pol))
    assert evaluate(pol, fs, asm) == evaluate_kleene(pol, fs, asm)


def test_kleene_incompleteness_on_nonlinear_policy_is_documented():
    pol = Or((Atom("p", "a"), Not(Atom("p", "a"))))
    assert not is_linear(pol)
    assert evaluate(pol, [], BASE_ASM) == Status.TRUE
    assert evaluate_kleene(pol, [], BASE_ASM) == Status.UNKNOWN  # sound-but-incomplete, expected


def test_strategy_covers_all_statuses():
    """Guard against a vacuous property suite: the generator must hit every status."""
    from hypothesis import find

    for target in Status:
        find(st.tuples(policies(), facts_lists(), assumptions()),
             lambda t, target=target: evaluate(*t) == target, settings=settings(max_examples=5000))


_ = (And, SUBJ, PREDS, ITEMS, replace)
