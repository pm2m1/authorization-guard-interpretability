"""Behavioural summary, Task A (oracle-status decoding) and Task B (error prediction) for the pilot.
Implements configs/pilot_config.json plus amendment AM1 (decision log D41; added before any behaviour or probe
result was inspected): Task A also gets structured and TF-IDF baselines, because oracle status is a deterministic
function of (family, condition) and decoding it is therefore only a sanity check.

All preprocessing (StandardScaler, TF-IDF) and C selection happen inside training folds (Pipeline + inner
GroupKFold on training scenarios). Outer CV = LeaveOneGroupOut over the 24 base scenarios.
Usage: uv run python scripts/probes.py [--perms-b 100] [--perms-a 20] > results/probe_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
BLOCKS = [7, 15, 23, 31]
REPS = ["last", "user_mean"]
STATUS = ["TRUE", "FALSE", "UNKNOWN", "INCONSISTENT"]
FAMS = ["existential", "universal", "cumulative"]
CONDS = ["POS", "NEG", "MISS", "MISS_LCWA", "MISS_IRREL", "CONFLICT", "MISS_CAP", "MISS_FLOOR"]


# ----------------------------------------------------------------------------------------------- data
def load():
    rows = [json.loads(line) for line in (HERE / "results/behaviour_full.jsonl").open()]
    rows = [r for r in rows if "item_id" in r]
    items = {json.loads(line)["item_id"]: json.loads(line) for line in (ROOT / "data/synthetic/exploratory_v1/items.jsonl").open()}
    reps = {}
    for f in sorted((HERE / "data/activations").glob("reps_*.npy")):
        ids = json.loads(f.with_suffix(".ids.json").read_text()) if f.with_suffix(".ids.json").exists() else \
            json.loads((f.parent / (f.stem + ".ids.json")).read_text())
        arr = np.load(f)
        for i, iid in enumerate(ids):
            reps[iid] = arr[i]
    for r in rows:
        r["user_prompt"] = items[r["item_id"]]["user_prompt"]
    return rows, reps


def structured(rows):
    X = []
    for r in rows:
        v = [r["family"] == f for f in FAMS] + [r["condition"] == c for c in CONDS]
        v += [r["completeness"] == "True", r["completeness"] == "False", float(r["pivot_rank"] or 0), r["n_prompt_tokens"] / 100.0]
        X.append(v)
    return np.array(X, dtype=float)


# ----------------------------------------------------------------------------------------------- models
class Cols(BaseEstimator, TransformerMixin):
    """Select a block of columns from a dict-of-arrays design (keeps TF-IDF fitting inside folds)."""

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class Design:
    """Fold-local feature builder: dense blocks are standardised; text gets a fold-fitted TF-IDF.

    Computational amendment AM2 (D42): a standardised dense block with more columns than training rows is
    re-expressed in the orthonormal basis of its TRAINING-row space (SVD of the standardised training block).
    For L2-penalised logistic regression this is an exact reparametrisation (the optimal weight vector lies in
    the span of the training rows; ||w|| = ||u|| for w = V u), so the estimator is unchanged; only runtime falls.
    The basis is computed from training rows only and applied to held-out rows.
    """

    def __init__(self, dense_keys, text=False, project=True):
        self.dense_keys, self.text, self.project = dense_keys, text, project

    def _dense(self, k, X):
        Z = self.scalers[k].transform(X)
        return Z @ self.bases[k].T if self.bases.get(k) is not None else Z

    def _stack(self, parts, text_part):
        if text_part is None:
            return np.hstack(parts)
        return sparse.hstack([sparse.csr_matrix(p) for p in parts] + [text_part]).tocsr()

    def fit_transform(self, D, idx):
        parts, self.scalers, self.bases = [], {}, {}
        for k in self.dense_keys:
            X = D[k][idx]
            s = StandardScaler().fit(X)
            self.scalers[k] = s
            if self.project and X.shape[1] > X.shape[0]:
                _, sv, vt = np.linalg.svd(s.transform(X), full_matrices=False)
                self.bases[k] = vt[sv > sv[0] * 1e-10]
            parts.append(self._dense(k, X))
        tp = None
        if self.text:
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([D["text"][i] for i in idx])
            tp = self.tfidf.transform([D["text"][i] for i in idx])
        return self._stack(parts, tp)

    def transform(self, D, idx):
        parts = [self._dense(k, D[k][idx]) for k in self.dense_keys]
        tp = self.tfidf.transform([D["text"][i] for i in idx]) if self.text else None
        return self._stack(parts, tp)


def fit_predict(D, y, groups, train, test, dense_keys, text, multiclass):
    """Nested: choose C by inner GroupKFold(4) on training scenarios; refit; return test scores."""
    inner = GroupKFold(n_splits=4)
    best, best_c = -np.inf, CS[0]
    for c in CS:
        sc = []
        for itr, iva in inner.split(train, y[train], groups[train]):
            tr, va = train[itr], train[iva]
            if len(set(y[tr])) < 2:
                continue
            des = Design(dense_keys, text)
            Xtr, Xva = des.fit_transform(D, tr), des.transform(D, va)
            m = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(Xtr, y[tr])
            if multiclass:
                sc.append(f1_score(y[va], m.predict(Xva), average="macro"))
            else:
                if len(set(y[va])) < 2:
                    continue
                sc.append(roc_auc_score(y[va], m.predict_proba(Xva)[:, 1]))
        if sc and np.mean(sc) > best:
            best, best_c = np.mean(sc), c
    des = Design(dense_keys, text)
    Xtr, Xte = des.fit_transform(D, train), des.transform(D, test)
    m = LogisticRegression(C=best_c, class_weight="balanced", max_iter=5000).fit(Xtr, y[train])
    return (m.predict(Xte) if multiclass else m.predict_proba(Xte)[:, 1]), best_c


def oof(D, y, groups, dense_keys, text, multiclass, n_jobs=-1):
    logo = LeaveOneGroupOut()
    splits = list(logo.split(np.zeros(len(y)), y, groups))
    for tr, te in splits:  # leakage check: scenario-disjoint
        assert not set(groups[tr]) & set(groups[te])
    res = Parallel(n_jobs=n_jobs)(delayed(fit_predict)(D, y, groups, tr, te, dense_keys, text, multiclass) for tr, te in splits)
    pred = np.empty(len(y), dtype=object if multiclass else float)
    cs = []
    for (tr, te), (p, c) in zip(splits, res):
        pred[te] = p
        cs.append(c)
    return pred, cs


# ----------------------------------------------------------------------------------------------- metrics
def boot_ci(groups, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(B):
        s = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by[g] for g in s])
        try:
            vals.append(fn(idx))
        except ValueError:
            continue
    return [round(float(np.quantile(vals, 0.025)), 3), round(float(np.quantile(vals, 0.975)), 3)]


def recall_at(y, s, frac):
    k = max(1, round(frac * len(y)))
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / max(1, y.sum()))


def risk_coverage(y, s):
    order = np.argsort(s, kind="stable")  # accept lowest-risk first
    out = {}
    for cov in (0.25, 0.5, 0.75, 0.9, 1.0):
        k = round(cov * len(y))
        out[str(cov)] = round(float(y[order[:k]].mean()), 3) if k else None
    return out


def binary_metrics(y, s, groups):
    d = {"auroc": round(float(roc_auc_score(y, s)), 3), "ap": round(float(average_precision_score(y, s)), 3),
             "base_rate": round(float(y.mean()), 3),
             "recall_at": {str(f): round(recall_at(y, s, f), 3) for f in (0.05, 0.10, 0.20)},
             "risk_coverage": risk_coverage(y, s)}
    d["auroc_ci"] = boot_ci(groups, lambda i: roc_auc_score(y[i], s[i]))
    d["ap_ci"] = boot_ci(groups, lambda i: average_precision_score(y[i], s[i]))
    return d


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms-b", type=int, default=100)
    ap.add_argument("--perms-a", type=int, default=20)
    ap.add_argument("--exclude-smoke", action="store_true")
    ap.add_argument("--save-oof", default=None, help="write out-of-fold Task B scores (JSON) for post-hoc analysis")
    a = ap.parse_args()
    rows, reps = load()
    t0 = time.time()
    oof_dump = {}
    out = {"label": "EXPLORATORY feasibility pilot (not confirmatory). Decodability != causal use.",
           "n_rows": len(rows), "exclude_smoke": a.exclude_smoke}
    for set_name, sel in (("primary_direct", lambda r: r["wording"] == "direct"), ("stress_dual", lambda r: r["wording"] == "dual")):
        R = [r for r in rows if sel(r) and not (a.exclude_smoke and r["smoke_item"])]
        groups = np.array([r["base_id"] for r in R])
        y_err = np.array([int(r["error"]) for r in R])
        S = {}
        # ---------------- behaviour
        conf = {g: {d: 0 for d in ["ALLOW", "DENY", "REQUEST_EVIDENCE", "ESCALATE"]} for g in ["ALLOW", "DENY", "REQUEST_EVIDENCE", "ESCALATE"]}
        for r in R:
            conf[r["gold"]][r["decision"]] += 1
        per_scen = {}
        for r in R:
            per_scen.setdefault(r["base_id"], 0)
            per_scen[r["base_id"]] += int(r["error"])
        cells = {}
        for r in R:
            k = f"{r['family']}|{r['condition']}"
            cells.setdefault(k, [0, 0])
            cells[k][0] += int(r["error"])
            cells[k][1] += 1
        S["behaviour"] = {"n": len(R), "errors": int(y_err.sum()), "accuracy": round(1 - y_err.mean(), 3),
                              "scenarios_with_errors": int(sum(v > 0 for v in per_scen.values())), "n_scenarios": len(per_scen),
                              "errors_per_scenario": per_scen, "errors_by_family_condition": {k: f"{v[0]}/{v[1]}" for k, v in sorted(cells.items())},
                              "confusion_gold_rows_decision_cols": conf,
                              "wrong_allow": int(sum(r["decision"] == "ALLOW" and r["gold"] != "ALLOW" for r in R))}
        D = {"text": [r["user_prompt"] for r in R], "struct": structured(R),
             "margin": np.array([[r["margin"]] for r in R]), "conf": np.array([[r["confidence"]] for r in R])}
        for bi, b in enumerate(BLOCKS):
            for ri, rp in enumerate(REPS):
                D[f"act_{b}_{rp}"] = np.stack([reps[r["item_id"]][bi, ri].astype(np.float32) for r in R])
        # ---------------- Task A: oracle status
        yA = np.array([r["gold_status"] for r in R], dtype=object)
        TA = {}
        specs_a = {"structured": (["struct"], False), "tfidf": ([], True)}
        specs_a |= {f"act_{b}_{rp}": ([f"act_{b}_{rp}"], False) for b in BLOCKS for rp in REPS}
        for name, (keys, text) in specs_a.items():
            pred, cs = oof(D, yA, groups, keys, text, True)
            corr = (pred == yA).astype(float)
            TA[name] = {"accuracy": round(float(corr.mean()), 3), "macro_f1": round(float(f1_score(yA, pred, average="macro")), 3),
                            "acc_ci": boot_ci(groups, lambda i, c=corr: c[i].mean()), "chosen_C": cs}
            if name.startswith("act_") and a.perms_a:
                rng = np.random.default_rng(1)
                perm = []
                for _ in range(a.perms_a):
                    yp = rng.permutation(yA)
                    pp, _ = oof(D, yp, groups, keys, text, True)
                    perm.append(f1_score(yp, pp, average="macro"))
                TA[name]["shuffled_macro_f1_mean"] = round(float(np.mean(perm)), 3)
                TA[name]["shuffled_macro_f1_max"] = round(float(np.max(perm)), 3)
            print(set_name, "A", name, TA[name]["macro_f1"], f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        S["taskA_oracle_status"] = TA
        # ---------------- Task B: error prediction
        TB = {}
        if y_err.sum() < 30:
            TB["stopped"] = f"fewer than 30 errors ({int(y_err.sum())}); pre-declared stopping rule"
        else:
            scores = {"confidence(1-max_p)": 1 - D["conf"][:, 0], "margin(-margin)": -D["margin"][:, 0]}
            specs_b = {"structured": (["struct"], False), "tfidf": ([], True), "text+margin": (["margin"], True),
                       "structured+margin": (["struct", "margin"], False)}
            specs_b |= {f"act_{b}_{rp}": ([f"act_{b}_{rp}"], False) for b in BLOCKS for rp in REPS}
            specs_b |= {f"act_{b}_{rp}+text+margin": ([f"act_{b}_{rp}", "margin"], True) for b in BLOCKS for rp in REPS}
            for name, (keys, text) in specs_b.items():
                s, cs = oof(D, y_err, groups, keys, text, False)
                scores[name] = s.astype(float)
                TB.setdefault("chosen_C", {})[name] = cs
                print(set_name, "B", name, round(roc_auc_score(y_err, scores[name]), 3), f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
            oof_dump[set_name] = {"item_id": [r["item_id"] for r in R], "error": y_err.tolist(),
                                  "scores": {k: [float(x) for x in v] for k, v in scores.items()}}
            TB["metrics"] = {k: binary_metrics(y_err, v, groups) for k, v in scores.items()}
            ref = scores["text+margin"]
            TB["incremental_vs_text+margin"] = {}
            for k, v in scores.items():
                if k.startswith("act_"):
                    TB["incremental_vs_text+margin"][k] = {
                        "d_auroc": round(float(roc_auc_score(y_err, v) - roc_auc_score(y_err, ref)), 3),
                        "d_auroc_ci": boot_ci(groups, lambda i, y=y_err, v=v, r=ref: roc_auc_score(y[i], v[i]) - roc_auc_score(y[i], r[i])),
                        "d_ap": round(float(average_precision_score(y_err, v) - average_precision_score(y_err, ref)), 3),
                        "d_ap_ci": boot_ci(groups, lambda i, y=y_err, v=v, r=ref: average_precision_score(y[i], v[i]) - average_precision_score(y[i], r[i]))}
            # within-stratum AUROC (pooled OOF scores) where >=10 errors and >=10 non-errors
            strata = {}
            gs = np.array([r["gold_status"] for r in R])
            for st in STATUS:
                m = gs == st
                if y_err[m].sum() >= 10 and (1 - y_err[m]).sum() >= 10:
                    strata[st] = {k: round(float(roc_auc_score(y_err[m], v[m])), 3) for k, v in scores.items()
                                  if k in ("text+margin", "margin(-margin)") or k.startswith("act_")}
                else:
                    strata[st] = f"insufficient (errors {int(y_err[m].sum())}, non-errors {int((1 - y_err[m]).sum())})"
            TB["within_status_auroc"] = strata
            # shuffled-label control for activation-only probes
            if a.perms_b:
                rng = np.random.default_rng(2)
                ctrl = {}
                for b in BLOCKS:
                    for rp in REPS:
                        key = f"act_{b}_{rp}"
                        vals = []
                        for _ in range(a.perms_b):
                            yp = rng.permutation(y_err)
                            sp, _ = oof(D, yp, groups, [key], False, False)
                            vals.append(roc_auc_score(yp, sp.astype(float)))
                        ctrl[key] = {"mean": round(float(np.mean(vals)), 3), "q95": round(float(np.quantile(vals, 0.95)), 3),
                                         "max": round(float(np.max(vals)), 3), "n": len(vals)}
                        print(set_name, "perm", key, ctrl[key], f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
                TB["shuffled_label_auroc"] = ctrl
        S["taskB_error_prediction"] = TB
        out[set_name] = S
    out["runtime_seconds"] = round(time.time() - t0, 1)
    if a.save_oof:
        Path(a.save_oof).write_text(json.dumps(oof_dump))
    json.dump(out, sys.stdout, indent=1, default=str)


if __name__ == "__main__":
    main()
