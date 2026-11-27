"""Tests for the label diagnostic helpers (selection balance, prompt variants, parser). No model needed."""
import json
from collections import Counter
from pathlib import Path

from mechinterp import label_diag as L

ROOT = Path(__file__).resolve().parents[3]


def _items():
    with (ROOT / "data/synthetic/exploratory_v1/items.jsonl").open() as fh:
        return [json.loads(line) for line in fh]


def test_selection_balance():
    sel = L.select_items(_items())
    assert len(sel) == 144 and len({s["item_id"] for s in sel}) == 144
    roles = Counter(s["diag_role"] for s in sel)
    assert roles == {"conflict": 72, "control_TRUE": 24, "control_FALSE": 24, "control_UNKNOWN": 24}
    assert all(s["wording"] == "direct" for s in sel)
    per_base = Counter(s["base_id"] for s in sel)
    assert len(per_base) == 24 and set(per_base.values()) == {6}
    fam = Counter((s["diag_role"], s["family"]) for s in sel if s["diag_role"] != "conflict")
    assert set(fam.values()) == {8} and len(fam) == 9


def test_prompt_variants_change_only_labels():
    orig = (ROOT / "data/synthetic/exploratory_v1/system_prompt.txt").read_text()
    assert L.system_prompt(orig, "A") == orig
    b = L.system_prompt(orig, "B")
    assert b.replace("NEEDS_REVIEW", "ESCALATE") == orig
    for c in "CD":
        p = L.system_prompt(orig, c)
        assert p.split("Decision rules:")[0] == orig.split("Decision rules:")[0]
        assert p.split("Evidence semantics:")[1].split("Respond")[0] == orig.split("Evidence semantics:")[1].split("Respond")[0]
        for dec, ident in L.CANDIDATES[c].items():
            assert f"- {ident} = {dec}: {L.RULE_TEXT[dec]}" in p
    # mappings: each decision gets different identifiers in C and D
    assert all(L.CANDIDATES["C"][d] != L.CANDIDATES["D"][d] for d in L.DECISIONS)


def test_parser():
    c = L.CANDIDATES["C"]
    assert L.parse_first_line("A\nThe evidence conflicts.", c) == ("ESCALATE", "STRICT")
    assert L.parse_first_line("**B**: satisfied", c) == ("ALLOW", "STRICT")
    assert L.parse_first_line("A = ESCALATE", c) == ("ESCALATE", "STRICT")
    assert L.parse_first_line("All owners approved", c) == (None, "INVALID")
    assert L.parse_first_line("NEEDS_REVIEW\nx", L.CANDIDATES["B"]) == ("ESCALATE", "STRICT")
    assert L.parse_first_line("", c) == (None, "EMPTY")
