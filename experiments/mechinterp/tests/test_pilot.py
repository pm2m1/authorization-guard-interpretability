"""Unit tests for the pilot's parser, metrics and fold hygiene (no model needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import probes as P

from mechinterp.model_io import LABELS, parse_decision


def test_parser_strict_and_invalid():
    assert parse_decision("ALLOW\nbecause") == ("ALLOW", "STRICT")
    assert parse_decision("**Decision: REQUEST_EVIDENCE**") == ("REQUEST_EVIDENCE", "STRICT")
    assert parse_decision("I think we should escalate") == ("ESCALATE", "LENIENT")
    assert parse_decision("ALLOW or DENY?")[1] == "AMBIGUOUS"
    assert parse_decision("no idea") == (None, "INVALID")
    assert parse_decision("") == (None, "EMPTY")
    assert len(LABELS) == 4


def test_recall_at_and_risk_coverage():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
    assert P.recall_at(y, s, 0.2) == 1.0
    assert P.recall_at(y, s, 0.1) == 0.5
    rc = P.risk_coverage(y, s)
    assert rc["0.5"] == 0.0 and rc["1.0"] == 0.2


def test_oof_is_scenario_disjoint_and_informative():
    rng = np.random.default_rng(0)
    n, g = 240, 12
    groups = np.repeat([f"S{i}" for i in range(g)], n // g)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 20))
    X[:, 0] += 2.5 * y
    D = {"x": X, "text": ["a b"] * n}
    s, cs = P.oof(D, y, groups, ["x"], False, False, n_jobs=1)
    assert len(cs) == g
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, s) > 0.85
    yp = rng.permutation(y)
    sp, _ = P.oof(D, yp, groups, ["x"], False, False, n_jobs=1)
    assert roc_auc_score(yp, sp) < 0.75


def test_design_fits_scaler_on_train_only():
    X = np.vstack([np.zeros((5, 2)), 100 * np.ones((5, 2))])
    des = P.Design(["x"])
    des.fit_transform({"x": X}, np.arange(5))
    assert np.allclose(des.scalers["x"].mean_, 0.0)


def test_row_space_projection_is_exact_reparametrisation():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    n, d = 60, 300
    X = rng.normal(size=(n + 20, d))
    y = (X[:, 0] + 0.5 * rng.normal(size=n + 20) > 0).astype(int)
    D = {"x": X}
    tr, te = np.arange(n), np.arange(n, n + 20)
    for C in (1e-3, 1.0):
        out = []
        for proj in (False, True):
            des = P.Design(["x"], project=proj)
            m = LogisticRegression(C=C, max_iter=20000, tol=1e-10).fit(des.fit_transform(D, tr), y[tr])
            out.append(m.predict_proba(des.transform(D, te))[:, 1])
        assert np.max(np.abs(out[0] - out[1])) < 1e-4
