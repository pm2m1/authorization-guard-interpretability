"""Possible-world reference oracle for authorization policies over incomplete evidence.

Semantics (see reports/02_candidate_a_formal_design.md):

  W(E) = all worlds over the finite atom set that satisfy (i) every evidence fact and
         (ii) the declared domain assumptions (locally-closed predicates, numeric domains).
  TRUE         iff W(E) non-empty and P holds in every world of W(E)
  FALSE        iff W(E) non-empty and P fails in every world of W(E)
  UNKNOWN      iff W(E) contains worlds satisfying P and worlds violating P
  INCONSISTENT iff W(E) is empty   (checked FIRST, so vacuous quantification never yields
                                    TRUE-and-FALSE)

The operational decision policy is a separate, explicit mapping (OPERATIONAL_POLICY).

Domain membership is assumed known (every quantifier ranges over an explicitly listed finite
roster). Unknown domain membership is out of scope for the exploratory study.

This module deliberately uses brute-force enumeration. An independent three-valued / interval
implementation lives in kleene.py and is cross-checked by property tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Literal

# ----------------------------------------------------------------------------- statuses


class Status(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"
    INCONSISTENT = "INCONSISTENT"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"
    ESCALATE = "ESCALATE"


# Controlled operational semantics for this experiment (not a claim about real systems).
OPERATIONAL_POLICY: dict[Status, Decision] = {
    Status.TRUE: Decision.ALLOW,
    Status.FALSE: Decision.DENY,
    Status.UNKNOWN: Decision.REQUEST_EVIDENCE,
    Status.INCONSISTENT: Decision.ESCALATE,
}


def operational_decision(status: Status) -> Decision:
    return OPERATIONAL_POLICY[status]


# ----------------------------------------------------------------------------- evidence

BoolAtom = tuple[str, str]  # (predicate, subject)
NumAtom = tuple[str, str]  # (attribute, item)


@dataclass(frozen=True)
class BoolFact:
    pred: str
    subject: str
    value: bool
    source: str = ""


@dataclass(frozen=True)
class NumFact:
    attr: str
    item: str
    op: Literal["eq", "ge", "le"]
    value: int
    source: str = ""


Fact = BoolFact | NumFact


@dataclass(frozen=True)
class NumDomain:
    """Finite grid lo, lo+step, ..., <=hi for every atom of one numeric attribute."""

    lo: int
    hi: int
    step: int = 1

    def values(self) -> range:
        return range(self.lo, self.hi + 1, self.step)


@dataclass(frozen=True)
class Assumptions:
    # Locally closed predicates (LCWA): for these, an atom not asserted TRUE is FALSE.
    complete_preds: frozenset[str] = frozenset()
    # Numeric domain per attribute (required for every numeric attribute used).
    num_domains: tuple[tuple[str, NumDomain], ...] = ()

    def domain(self, attr: str) -> NumDomain:
        for a, d in self.num_domains:
            if a == attr:
                return d
        raise KeyError(f"no numeric domain declared for attribute {attr!r}")


# ----------------------------------------------------------------------------- policy AST


@dataclass(frozen=True)
class Atom:
    pred: str
    subject: str  # may be the bound variable name, e.g. "$x"


@dataclass(frozen=True)
class Not:
    arg: Policy


@dataclass(frozen=True)
class And:
    args: tuple[Policy, ...]


@dataclass(frozen=True)
class Or:
    args: tuple[Policy, ...]


@dataclass(frozen=True)
class Exists:
    var: str
    domain: tuple[str, ...]
    body: Policy


@dataclass(frozen=True)
class Forall:
    var: str
    domain: tuple[str, ...]
    body: Policy


@dataclass(frozen=True)
class SumCmp:
    """sum(attr(i) for i in items) + extra  <op>  limit, op in {'le','gt'}."""

    attr: str
    items: tuple[str, ...]
    extra: int
    op: Literal["le", "gt"]
    limit: int


Policy = Atom | Not | And | Or | Exists | Forall | SumCmp


def substitute(p: Policy, var: str, value: str) -> Policy:
    if isinstance(p, Atom):
        return Atom(p.pred, value if p.subject == var else p.subject)
    if isinstance(p, Not):
        return Not(substitute(p.arg, var, value))
    if isinstance(p, And):
        return And(tuple(substitute(a, var, value) for a in p.args))
    if isinstance(p, Or):
        return Or(tuple(substitute(a, var, value) for a in p.args))
    if isinstance(p, Exists | Forall):
        if p.var == var:  # shadowed
            return p
        return type(p)(p.var, p.domain, substitute(p.body, var, value))
    if isinstance(p, SumCmp):
        return p
    raise TypeError(p)


def expand(p: Policy) -> Policy:
    """Expand quantifiers over their finite domains into And/Or of ground formulas."""
    if isinstance(p, Atom | SumCmp):
        return p
    if isinstance(p, Not):
        return Not(expand(p.arg))
    if isinstance(p, And):
        return And(tuple(expand(a) for a in p.args))
    if isinstance(p, Or):
        return Or(tuple(expand(a) for a in p.args))
    if isinstance(p, Exists):
        return Or(tuple(expand(substitute(p.body, p.var, d)) for d in p.domain))
    if isinstance(p, Forall):
        return And(tuple(expand(substitute(p.body, p.var, d)) for d in p.domain))
    raise TypeError(p)


def atoms_of(p: Policy) -> tuple[set[BoolAtom], set[NumAtom]]:
    b: set[BoolAtom] = set()
    n: set[NumAtom] = set()

    def walk(q: Policy) -> None:
        if isinstance(q, Atom):
            if q.subject.startswith("$"):
                raise ValueError("unbound variable in ground formula")
            b.add((q.pred, q.subject))
        elif isinstance(q, Not):
            walk(q.arg)
        elif isinstance(q, And | Or):
            for a in q.args:
                walk(a)
        elif isinstance(q, SumCmp):
            for i in q.items:
                n.add((q.attr, i))
        else:
            raise TypeError(q)

    walk(expand(p))
    return b, n


def holds(p: Policy, world_b: dict[BoolAtom, bool], world_n: dict[NumAtom, int]) -> bool:
    if isinstance(p, Atom):
        return world_b[(p.pred, p.subject)]
    if isinstance(p, Not):
        return not holds(p.arg, world_b, world_n)
    if isinstance(p, And):
        return all(holds(a, world_b, world_n) for a in p.args)
    if isinstance(p, Or):
        return any(holds(a, world_b, world_n) for a in p.args)
    if isinstance(p, SumCmp):
        total = sum(world_n[(p.attr, i)] for i in p.items) + p.extra
        return total <= p.limit if p.op == "le" else total > p.limit
    raise TypeError(p)  # quantifiers must be expanded first


# ----------------------------------------------------------------------------- enumeration


def _bool_candidates(atom: BoolAtom, facts: list[Fact], asm: Assumptions) -> list[bool]:
    asserted = {f.value for f in facts if isinstance(f, BoolFact) and (f.pred, f.subject) == atom}
    cands = [False, True]
    cands = [v for v in cands if all(v == a for a in asserted)]  # must satisfy every assertion
    if atom[0] in asm.complete_preds:
        # LCWA: atom is TRUE exactly when it is asserted TRUE.
        closed = True in asserted
        cands = [v for v in cands if v == closed]
    return cands


def _num_candidates(atom: NumAtom, facts: list[Fact], asm: Assumptions) -> list[int]:
    out = []
    rel = [f for f in facts if isinstance(f, NumFact) and (f.attr, f.item) == atom]
    for v in asm.domain(atom[0]).values():
        ok = True
        for f in rel:
            if f.op == "eq" and v != f.value:
                ok = False
            elif f.op == "ge" and v < f.value:
                ok = False
            elif f.op == "le" and v > f.value:
                ok = False
        if ok:
            out.append(v)
    return out


def possible_worlds(policy: Policy, facts: list[Fact], asm: Assumptions):
    """Yield (bool_world, num_world) for every world in W(E) over atoms of policy + evidence."""
    pb, pn = atoms_of(policy)
    eb = {(f.pred, f.subject) for f in facts if isinstance(f, BoolFact)}
    en = {(f.attr, f.item) for f in facts if isinstance(f, NumFact)}
    batoms = sorted(pb | eb)
    natoms = sorted(pn | en)
    bc = [_bool_candidates(a, facts, asm) for a in batoms]
    nc = [_num_candidates(a, facts, asm) for a in natoms]
    for bv in product(*bc):
        wb = dict(zip(batoms, bv, strict=True))
        for nv in product(*nc):
            yield wb, dict(zip(natoms, nv, strict=True))


def evaluate(policy: Policy, facts: list[Fact], asm: Assumptions) -> Status:
    """Reference epistemic status by exhaustive possible-world enumeration."""
    ground = expand(policy)
    seen_sat = seen_unsat = False
    any_world = False
    for wb, wn in possible_worlds(policy, facts, asm):
        any_world = True
        if holds(ground, wb, wn):
            seen_sat = True
        else:
            seen_unsat = True
        if seen_sat and seen_unsat:
            return Status.UNKNOWN
    if not any_world:
        return Status.INCONSISTENT
    return Status.TRUE if seen_sat else Status.FALSE


def count_worlds(policy: Policy, facts: list[Fact], asm: Assumptions) -> tuple[int, int]:
    """(#worlds satisfying P, #worlds violating P) — used for diagnostics only."""
    ground = expand(policy)
    s = u = 0
    for wb, wn in possible_worlds(policy, facts, asm):
        if holds(ground, wb, wn):
            s += 1
        else:
            u += 1
    return s, u
