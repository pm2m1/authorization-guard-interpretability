"""Deserialise the `formal` block stored with each item back into oracle objects."""

from __future__ import annotations

from . import oracle as O

_T = {c.__name__: c for c in (O.Atom, O.Not, O.And, O.Or, O.Exists, O.Forall, O.SumCmp, O.BoolFact,
                              O.NumFact, O.NumDomain, O.Assumptions)}


def _de(x):
    if isinstance(x, dict) and "_type" in x:
        cls = _T[x["_type"]]
        kw = {k: _de(v) for k, v in x.items() if k != "_type"}
        if cls is O.Assumptions:
            kw["complete_preds"] = frozenset(kw["complete_preds"])
            kw["num_domains"] = tuple((a, d) for a, d in kw["num_domains"])
        for k in ("args", "domain", "items"):
            if k in kw and isinstance(kw[k], list):
                kw[k] = tuple(kw[k])
        return cls(**kw)
    if isinstance(x, list):
        return [_de(v) for v in x]
    return x


def formal_from_item(item: dict):
    f = item["formal"]
    return _de(f["policy"]), _de(f["facts"]), _de(f["assumptions"])
