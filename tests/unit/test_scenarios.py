"""Validity checks on the frozen exploratory item set (design balance, labels, leakage)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from authorization_closure.generate import FORBIDDEN_IN_PROMPTS, INTENDED, N_LINES, build_items
from authorization_closure.kleene import evaluate_kleene
from authorization_closure.oracle import Status, evaluate, operational_decision
from authorization_closure.serialization import formal_from_item

P = Path(__file__).resolve().parents[2] / "data/synthetic/exploratory_v1/items.jsonl"


@pytest.fixture(scope="module")
def items():
    if not P.exists():
        pytest.skip("frozen item set not generated")
    return [json.loads(line) for line in P.open()]


def test_frozen_file_matches_generator(items):
    """The frozen file is exactly what the current generator produces (no silent drift)."""
    regenerated = build_items()
    key = lambda d: d["item_id"]  # noqa: E731
    assert sorted(items, key=key) == sorted(json.loads(json.dumps(regenerated)), key=key)


def test_cell_balance(items):
    c = Counter((i["family"], i["condition"], i["wording"]) for i in items)
    assert len(items) == 864 and set(c.values()) == {24}


def test_labels_recomputed_from_formal_representation(items):
    for it in items:
        pol, facts, asm = formal_from_item(it)
        st = evaluate(pol, facts, asm)
        assert st.value == it["oracle_status"] == INTENDED[(it["family"], it["condition"])].value
        assert evaluate_kleene(pol, facts, asm) == st
        assert operational_decision(st).value == it["oracle_decision"]


def test_no_unknown_or_inconsistent_item_is_authorized_by_oracle(items):
    for it in items:
        if it["oracle_status"] in ("UNKNOWN", "INCONSISTENT"):
            assert it["oracle_decision"] != "ALLOW"


def test_constant_evidence_line_count(items):
    for it in items:
        body = it["user_prompt"].split("Evidence:\n")[1].split("\n\n")[0]
        assert len(body.splitlines()) == N_LINES == it["n_evidence_lines"]


def test_no_label_vocabulary_or_design_codes_in_prompts(items):
    for it in items:
        low = it["user_prompt"].lower()
        assert not [w for w in FORBIDDEN_IN_PROMPTS if w in low]
        for code in ("MISS", "NEG", "POS", "CONFLICT", "LCWA", "IRREL"):
            assert not re.search(rf"\b{code}\b", it["user_prompt"])
        assert it["item_id"] not in it["user_prompt"]


def test_item_ids_unique_and_uninformative(items):
    ids = [i["item_id"] for i in items]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[0-9a-f]{10}", x) for x in ids)


def test_completeness_declaration_is_not_a_label_cue(items):
    labels_with_decl = Counter(i["oracle_status"] for i in items if i["completeness_declared"])
    assert len(labels_with_decl) >= 3  # appears with TRUE, FALSE and INCONSISTENT, not only FALSE


def test_cumulative_ceiling_line_present_in_every_cumulative_item(items):
    for it in items:
        if it["family"] == "cumulative":
            assert "booked with a ceiling of" in it["user_prompt"]


def test_prompt_length_roughly_matched_within_family(items):
    by = defaultdict(list)
    for it in items:
        by[(it["family"], it["condition"])].append(it["n_chars_user"])
    for fam in ("existential", "universal", "cumulative"):
        means = [sum(v) / len(v) for (f, _), v in by.items() if f == fam]
        assert max(means) / min(means) < 1.15, fam


def test_minimal_pairs_share_non_pivot_records(items):
    """Within (base, family, wording) every record line about a NON-pivot subject is identical
    across the POS/NEG/MISS/CONFLICT conditions (MISS_IRREL deliberately changes one)."""
    groups = defaultdict(dict)
    for it in items:
        groups[(it["base_id"], it["family"], it["wording"])][it["condition"]] = it
    for (_, fam, _), g in groups.items():
        if fam == "cumulative":
            continue
        piv = g["POS"]["pivot_subject"]

        def recs(it, piv=piv):
            body = it["user_prompt"].split("Evidence:\n")[1]
            return sorted(line for line in body.splitlines() if " — " in line and piv not in line)

        ref = recs(g["POS"])
        for c in ("NEG", "MISS", "CONFLICT", "MISS_LCWA"):
            assert recs(g[c]) == ref


def test_status_space_covered_per_family(items):
    for fam in ("existential", "universal", "cumulative"):
        assert {i["oracle_status"] for i in items if i["family"] == fam} == {s.value for s in Status}
