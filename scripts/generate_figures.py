"""Generate the repository figures (SVG) from preserved result files. Standard library only; no model, no inference.

Usage (from the repository root):
    python3 scripts/generate_figures.py            # writes figures/*.svg
    python3 scripts/generate_figures.py --png      # also writes PNGs if `rsvg-convert` is installed (optional)

Every plotted number is read from experiments/mechinterp/{results,configs,manifests} or data/synthetic. The output is
deterministic (no timestamps), so re-running on unchanged inputs reproduces the files byte for byte.
Colors: colorblind-validated categorical slots (blue / orange / aqua), with light and dark variants selected by
`prefers-color-scheme` inside each SVG.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MI = ROOT / "experiments/mechinterp"
OUT = ROOT / "figures"
LABELS = ("ALLOW", "DENY", "REQUEST_EVIDENCE", "ESCALATE")

# ----------------------------------------------------------------------------------------------------------- data


def load_json(rel: str):
    return json.loads((MI / rel).read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def argmax(scores: dict) -> str:
    return max(LABELS, key=lambda k: scores[k])


def data() -> dict:
    beh = [r for r in load_jsonl(MI / "results/behaviour_full.jsonl") if "item_id" in r]
    raw = {r["item_id"]: r for r in beh}
    diag_items = load_jsonl(MI / "manifests/label_diag/items.jsonl")
    conflict = [raw[i["item_id"]] for i in diag_items if i["diag_role"] == "conflict"]
    assert all(argmax(r["label_logprob"]) == r["decision"] for r in beh)
    gold = {r["gold"] for r in conflict}
    assert gold == {"ESCALATE"}, gold
    src = load_jsonl(ROOT / "data/synthetic/exploratory_v1/items.jsonl")

    audit = load_json("results/label_diag/audit_taskA.json")
    posthoc = load_json("results/posthoc_taskA_per_class.json")
    recall, counts = {}, {}
    for spec in ["tfidf"] + [f"act_{b}_{p}" for b in (7, 15, 23, 31) for p in ("last", "user_mean")]:
        rec = audit[spec]["per_class"]["INCONSISTENT"]["recall"]
        assert abs(rec - posthoc[spec]["recall"]["INCONSISTENT"]) < 1e-9, spec  # two saved analyses must agree
        recall[spec] = rec
        counts[spec] = (audit[spec]["confusion_gold_rows"]["INCONSISTENT"]["INCONSISTENT"],
                        audit[spec]["per_class"]["INCONSISTENT"]["support"])

    tb = load_json("results/probe_results.json")["primary_direct"]["taskB_error_prediction"]["metrics"]
    acts = [k for k in tb if k.startswith("act_") and "+" not in k]
    best = max(acts, key=lambda k: tb[k]["auroc"])

    pilot = load_json("configs/pilot_config.json")
    cfg = load_json("manifests/config.json")
    text_cfg = cfg.get("text_config", cfg)
    return {
        "n_items": len(beh), "n_bases": len({r["base_id"] for r in src}),
        "n_direct": sum(r["wording"] == "direct" for r in beh), "n_dual": sum(r["wording"] == "dual" for r in beh),
        "n_diag": len(diag_items), "n_diag_conflict": len(conflict),
        "conflict_counts": {lab: sum(argmax(r["label_logprob"]) == lab for r in conflict) for lab in LABELS},
        "recall": recall, "recall_counts": counts,
        "taskB": tb, "best_act": best, "act_range": (min(tb[k]["auroc"] for k in acts), max(tb[k]["auroc"] for k in acts)),
        "n_acts": len(acts),
        "diag": load_json("results/label_diag/analysis.json"), "freegen": load_json("results/label_diag/freegen_by_norm.json"),
        "blocks": pilot["extraction"]["blocks_0_indexed"], "hidden": text_cfg["hidden_size"],
        "layer_types": text_cfg["layer_types"], "n_layers": text_cfg["num_hidden_layers"],
        "seed": json.loads((ROOT / "manifests/exploratory_v1_dataset.json").read_text())["generation_command"].split()[-1],
    }


# ----------------------------------------------------------------------------------------------------------- svg

STYLE = """<style>
text{font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;fill:#0b0b0b}
.bg{fill:#fcfcfb}.title{font-size:17px;font-weight:600}.sub{font-size:13px;fill:#52514e}
.t{font-size:13px}.s{font-size:12px;fill:#52514e}.b{font-weight:600}.mut{fill:#898781}.ink2{fill:#52514e}
.grid{stroke:#e1e0d9;stroke-width:1}.axis{stroke:#c3c2b7;stroke-width:1}.ref{stroke:#52514e;stroke-width:1.5}
.s1{fill:#2a78d6}.s2{fill:#eb6834}.s3{fill:#1baf7a}.l1{stroke:#2a78d6}.l2{stroke:#eb6834}.l3{stroke:#1baf7a}
.ring{stroke:#fcfcfb;stroke-width:2}.outline{fill:none;stroke:#52514e;stroke-width:1.5}
.box{fill:#f3f2ee;stroke:#c3c2b7;stroke-width:1}.boxa{fill:#eaf1fb;stroke:#2a78d6;stroke-width:1.2}
.boxb{fill:#fdeee7;stroke:#eb6834;stroke-width:1.2}.boxc{fill:#e6f5ef;stroke:#1baf7a;stroke-width:1.2}
.cellf{fill:#2a78d6}.celll{fill:#e1e0d9}.arrow{stroke:#52514e;stroke-width:1.5;fill:none}.head{fill:#52514e}
.hatch{stroke:#2a78d6;stroke-width:2}.hatchbg{fill:#fcfcfb}.sep{stroke:#c3c2b7;stroke-width:1}
@media (prefers-color-scheme: dark){
text{fill:#f0efec}.bg{fill:#1a1a19}.sub,.s,.ink2{fill:#c3c2b7}.grid{stroke:#2c2c2a}.axis{stroke:#383835}.ref{stroke:#c3c2b7}
.s1{fill:#3987e5}.s2{fill:#d95926}.s3{fill:#199e70}.l1{stroke:#3987e5}.l2{stroke:#d95926}.l3{stroke:#199e70}
.ring{stroke:#1a1a19}.outline{stroke:#c3c2b7}.box{fill:#242422;stroke:#4a4a46}.boxa{fill:#1b2a3d;stroke:#3987e5}
.boxb{fill:#3a2219;stroke:#d95926}.boxc{fill:#16302a;stroke:#199e70}.cellf{fill:#3987e5}.celll{fill:#383835}
.arrow{stroke:#c3c2b7}.head{fill:#c3c2b7}.hatch{stroke:#3987e5}.hatchbg{fill:#1a1a19}.sep{stroke:#383835}}
</style>"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def doc(w: int, h: int, title: str, desc: str, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-labelledby="t d">\n<title id="t">{esc(title)}</title>\n<desc id="d">{esc(desc)}</desc>\n{STYLE}\n'
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            '<path class="head" d="M0,0 L10,5 L0,10 z"/></marker>'
            '<pattern id="hatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">'
            '<rect class="hatchbg" width="6" height="6"/><line class="hatch" x1="0" y1="0" x2="0" y2="6"/></pattern></defs>\n'
            f'<rect class="bg" width="{w}" height="{h}" rx="8"/>\n' + "\n".join(body) + "\n</svg>\n")


def T(x, y, s, cls="t", anchor="start", extra="") -> str:
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}"{a}{extra}>{esc(s)}</text>'


def line(x1, y1, x2, y2, cls) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>'


def seg(x1, y1, x2, y2, cls, width=2.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" stroke-width="{width}"/>'


def hbar(x, y, w, h, cls, r=4) -> str:
    """Horizontal bar anchored at x with a rounded data end (right side only)."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f'<path class="{cls}" d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z"/>')


def vbar(x, y0, w, h, cls, r=4) -> str:
    """Vertical bar anchored at baseline y0 with a rounded data end (top only)."""
    if h <= 0:
        return ""
    r = min(r, h, w / 2)
    top = y0 - h
    return (f'<path class="{cls}" d="M{x:.1f},{y0:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} '
            f'H{x + w - r:.1f} Q{x + w:.1f},{top:.1f} {x + w:.1f},{top + r:.1f} V{y0:.1f} Z"/>')


def box(x, y, w, h, lines: list[str], cls="box", size=13, lead=18) -> list[str]:
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="{cls}"/>']
    for i, s in enumerate(lines):
        out.append(T(x + 12, y + 22 + i * lead, s, "t b" if i == 0 else "s", extra=f' style="font-size:{size if i == 0 else size - 1}px"'))
    return out


def arrow(points: list[tuple[float, float]], label: str | None = None, lx=None, ly=None) -> list[str]:
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
    out = [f'<path class="arrow" d="{d}" marker-end="url(#ah)"/>']
    if label:
        out.append(T(lx, ly, label, "s"))
    return out


def pct(k, n) -> str:
    return f"{100 * k / n:.0f}%"


# ----------------------------------------------------------------------------------------------------------- charts


def fig_conflict(d) -> str:
    c, n = d["conflict_counts"], d["n_diag_conflict"]
    W, x0, x1, top, rowh, bh = 720, 200, 640, 112, 44, 20
    sc = (x1 - x0) / n
    b = [T(20, 30, f"Original-interface judgments on the {n} contradictory-evidence items", "title"),
         T(20, 50, "Qwen3.5-9B bf16, forced choice over four labels; direct-wording CONFLICT items (oracle status INCONSISTENT)", "sub"),
         '<rect x="20" y="66" width="14" height="10" class="s1"/>', T(40, 75, "Observed model judgment", "s"),
         '<rect x="230" y="66" width="14" height="10" class="outline"/>', T(250, 75, f"Oracle-expected decision (ESCALATE for all {n} items)", "s")]
    ticks = [round(n * k / 4) for k in range(5)]
    for t in ticks:
        x = x0 + t * sc
        b += [line(x, top - 8, x, top + 4 * rowh - 12, "grid"), T(x, top + 4 * rowh + 6, t, "s", "middle")]
    b.append(T((x0 + x1) / 2, top + 4 * rowh + 26, f"Number of items (of {n})", "s", "middle"))
    for i, lab in enumerate(LABELS):
        y = top + i * rowh
        b.append(T(x0 - 12, y + bh - 5, lab, "t", "end"))
        if lab == "ESCALATE":
            b.append(f'<rect x="{x0}" y="{y - 3:.1f}" width="{n * sc:.1f}" height="{bh + 6}" rx="4" class="outline"/>')
        b.append(hbar(x0, y, c[lab] * sc, bh, "s1"))
        lx = x0 + c[lab] * sc + 8 if lab != "ESCALATE" else x0 + 8
        b.append(T(lx, y + bh - 5, f"{c[lab]} ({pct(c[lab], n)})" if lab != "ESCALATE" else f"0 observed; expected {n}", "t b"))
        b.append(line(x0, y - 4, x0, y + bh + 4, "axis"))
    yb = top + 4 * rowh + 52
    b += [T(20, yb, "These are text-level authorization judgments scored from label log-probabilities; no tool or action was executed.", "s"),
          T(20, yb + 18, "Source: experiments/mechinterp/results/behaviour_full.jsonl (decision = argmax of stored log-probs),", "s mut"),
          T(20, yb + 34, "items from manifests/label_diag/items.jsonl. ESCALATE was chosen on 0 of all "
            f"{d['n_items']} items.", "s mut")]
    return doc(W, yb + 52, "Original-interface decisions on contradictory-evidence items",
               f"Of {n} CONFLICT items whose oracle decision is ESCALATE, the model chose ALLOW {c['ALLOW']}, DENY {c['DENY']}, "
               f"REQUEST_EVIDENCE {c['REQUEST_EVIDENCE']} and ESCALATE {c['ESCALATE']} times.", b)


def fig_layerwise(d) -> str:
    rec, cnt, blocks = d["recall"], d["recall_counts"], d["blocks"]
    W, xl, xr, yt, yb = 780, 150, 560, 96, 336
    xs = {blk: xl + (xr - xl) * i / (len(blocks) - 1) for i, blk in enumerate(blocks)}

    def Y(v):
        return yb - (yb - yt) * v

    b = [T(20, 30, "Linear decodability of the CONFLICT construction (INCONSISTENT recall)", "title"),
         T(20, 50, "POST-HOC, EXPLORATORY · 4-class probe, leave-one-scenario-out over 24 scenarios, primary set (n = 432)", "sub")]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        b += [line(xl - 40, Y(v), xr + 20, Y(v), "grid"), T(xl - 48, Y(v) + 4, f"{v:.2f}", "s", "end")]
    b.append(T(24, (yt + yb) / 2, "INCONSISTENT recall (n = 72 per probe)", "s", "middle",
               f' transform="rotate(-90 24 {(yt + yb) / 2:.1f})"'))
    for blk in blocks:
        b.append(T(xs[blk], yb + 22, f"Block {blk}" + (" (pre-norm)" if blk == blocks[-1] else ""), "s", "middle"))
    b.append(T((xl + xr) / 2, yb + 44, "Sampled decoder block (0-indexed; all are full-attention blocks)", "s", "middle"))
    tf = rec["tfidf"]
    b += [line(xl - 40, Y(tf), xr + 20, Y(tf), "ref"), T(xr + 28, Y(tf) + 4, f"TF-IDF baseline {tf:.3f}", "s")]
    series = [("last", "l1", "s1", "Last prompt token"), ("user_mean", "l2", "s2", "Mean over user tokens")]
    for pos, lc, _fc, _name in series:
        pts = " ".join(f"{xs[blk]:.1f},{Y(rec[f'act_{blk}_{pos}']):.1f}" for blk in blocks)
        b.append(f'<polyline points="{pts}" fill="none" class="{lc}" stroke-width="2"/>')
    for blk in blocks:
        vals = {pos: rec[f"act_{blk}_{pos}"] for pos, *_ in series}
        hi = max(vals, key=vals.get)
        for pos, _lc, fc, _ in series:
            v, x, y = vals[pos], xs[blk], Y(vals[pos])
            mark = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" class="{fc} ring"/>' if pos == "last" else
                    f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" class="{fc} ring"/>')
            k, n = cnt[f"act_{blk}_{pos}"]
            b += [mark, T(x, y - 12 if pos == hi else y + 21, f"{v:.3f} ({k}/{n})", "s", "middle")]
    lx = xr + 28
    yl = {pos: Y(rec[f"act_{blocks[-1]}_{pos}"]) for pos, *_ in series}
    if abs(yl["last"] - yl["user_mean"]) < 18:
        yl["user_mean"] = yl["last"] + 18
    for pos, _lc, fc, name in series:
        sw = (f'<circle cx="{lx + 5}" cy="{yl[pos]:.1f}" r="5" class="{fc}"/>' if pos == "last"
              else f'<rect x="{lx}" y="{yl[pos] - 5:.1f}" width="10" height="10" class="{fc}"/>')
        b += [sw, T(lx + 16, yl[pos] + 4, name, "t")]
    lo = min(v for k, v in rec.items() if k != "tfidf")
    hi_ = max(v for k, v in rec.items() if k != "tfidf")
    yc = yb + 76
    b += [T(20, yc, f"All 8 probes: {lo:.3f}–{hi_:.3f}. Declared after the main results were seen; the construction-aware baseline decodes", "s"),
          T(20, yc + 18, "INCONSISTENT perfectly only via the condition label (F1 ≤ 0.182 without it). Decodability is correlational, not causal use.", "s"),
          T(20, yc + 36, "Source: experiments/mechinterp/results/label_diag/audit_taskA.json (cross-checked against posthoc_taskA_per_class.json)", "s mut")]
    return doc(W, yc + 54, "Layer-wise INCONSISTENT recall (post-hoc)",
               "INCONSISTENT recall for blocks " + ", ".join(map(str, blocks)) + " and two positions: " +
               "; ".join(f"{k} {v:.3f}" for k, v in rec.items()), b)


def fig_error(d) -> str:
    tb, best = d["taskB"], d["best_act"]
    blk, pos = best.split("_")[1], "last token" if best.endswith("last") else "user-token mean"
    rows = [("Deployable", None, None), ("Output margin", "margin(-margin)", "s1"), ("TF-IDF (user prompt)", "tfidf", "s1"),
            ("Text + margin", "text+margin", "s1"),
            ("Activation probe", None, None), (f"Best of {d['n_acts']}: block {blk}, {pos}", best, "s2"),
            ("Construction-aware (diagnostic ceiling, not deployable)", None, None),
            ("Structured features", "structured", "s3"), ("Structured + margin", "structured+margin", "s3")]
    W, xl, xr, top = 800, 250, 590, 92
    lo, hi = 0.5, 1.0

    def X(v):
        return xl + (xr - xl) * (v - lo) / (hi - lo)

    b = [T(20, 30, "Error prediction: activations versus cheaper and construction-aware signals", "title"),
         T(20, 50, "AUROC for predicting the model's own authorization errors; primary set n = 432 (203 errors); "
           "95% scenario-cluster bootstrap intervals", "sub")]
    y, ys = top, []
    for name, key, cls in rows:
        if key is None:
            y += 6
            b.append(T(20, y + 4, name, "t b"))
            y += 22
            continue
        ys.append(y)
        m = tb[key]
        l_, u_ = m["auroc_ci"]
        lc = "l" + cls[1:]
        b += [T(xl - 14, y + 4, name, "t", "end"), seg(X(l_), y, X(u_), y, lc),
              seg(X(l_), y - 5, X(l_), y + 5, lc), seg(X(u_), y - 5, X(u_), y + 5, lc),
              f'<circle cx="{X(m["auroc"]):.1f}" cy="{y:.1f}" r="6" class="{cls} ring"/>',
              T(xr + 16, y + 4, f'{m["auroc"]:.3f} [{l_:.3f}, {u_:.3f}]', "t")]
        y += 30
    yb = y - 6
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        b.insert(2, line(X(v), top - 12, X(v), yb, "grid"))
        b.append(T(X(v), yb + 18, f"{v:.1f}", "s", "middle"))
    b += [T(X(0.5) + 4, top - 2, "chance", "s mut"), T((xl + xr) / 2, yb + 38, "AUROC (higher = better error ranking)", "s", "middle")]
    a0, a1 = d["act_range"]
    yc = yb + 66
    b += [T(20, yc, "Pre-declared conclusion: activations beat text + margin but not the construction-aware features, and they show no", "s"),
          T(20, yc + 18, "error-ranking signal within cells where structure is fixed, so no credible value beyond item structure. "
            f"All {d['n_acts']} activation probes: {a0:.3f}–{a1:.3f}.", "s"),
          T(20, yc + 36, "Structured features encode the dataset construction (including the condition that defines CONFLICT); they are not available in deployment.", "s"),
          T(20, yc + 54, "Source: experiments/mechinterp/results/probe_results.json → primary_direct.taskB_error_prediction.metrics", "s mut")]
    return doc(W, yc + 72, "Error-prediction AUROC by score",
               "; ".join(f"{name}: {tb[k]['auroc']:.3f}" for name, k, _ in rows if k), b)


def fig_interface(d) -> str:
    A, fg = d["diag"], d["freegen"]
    conds = [("A", "Original labels"), ("B", "ESCALATE → NEEDS_REVIEW"), ("C", "Letters, A = ESCALATE"), ("D", "Letters, C = ESCALATE")]
    norms = [("norm_total", "Total log-prob (primary)", "s1"), ("norm_first", "First token (sensitivity)", "s2")]
    n = A["norm_total"]["A"]["conflict_n"]
    W, xl, xr, yt, yb, bw = 760, 80, 720, 100, 300, 34
    gw = (xr - xl) / len(conds)

    def H(v):
        return (yb - yt) * v / n

    b = [T(20, 30, "Escalation depends on the answer interface and on how answers are scored", "title"),
         T(20, 50, f"Correct escalations on {n} CONFLICT items (behaviour only; no activations, no intervention)", "sub")]
    lx = 20
    for _k, name, cls in norms:
        b += [f'<rect x="{lx}" y="64" width="14" height="10" class="{cls}"/>', T(lx + 20, 73, name, "s")]
        lx += 200
    b += [f'<rect x="{lx}" y="64" width="14" height="10" fill="url(#hatch)"/>', T(lx + 20, 73, "Fails the free-generation agreement check", "s")]
    for t in range(0, n + 1, n // 4):
        b += [line(xl, yb - H(t), xr, yb - H(t), "grid"), T(xl - 8, yb - H(t) + 4, t, "s", "end")]
    b.append(T(24, (yt + yb) / 2, f"Correct escalations (of {n})", "s", "middle", f' transform="rotate(-90 24 {(yt + yb) / 2:.1f})"'))
    for i, (c, name) in enumerate(conds):
        gx = xl + i * gw + gw / 2
        for j, (nk, _nm, cls) in enumerate(norms):
            v = A[nk][c]["conflict_escalations"]
            agree, ng = fg[c]["agree_by_norm"]["total" if nk == "norm_total" else "first"], fg[c]["n"]
            x = gx - bw - 1 + j * (bw + 2)
            if agree < ng:  # failed the free-generation agreement check: hatched, labelled, never solid
                b.append(vbar(x, yb, bw, H(v), "hatchfill").replace('class="hatchfill"', 'fill="url(#hatch)"'))
                b.append(T(x + bw / 2, yb - H(v) - 22, "fails check", "s b", "middle"))
            else:
                b.append(vbar(x, yb, bw, H(v), cls))
            b.append(T(x + bw / 2, yb - H(v) - 6, v, "t b", "middle"))
        b += [T(gx, yb + 20, f"{c}", "t b", "middle"), T(gx, yb + 38, name, "s", "middle")]
    b.append(line(xl, yb, xr, yb, "axis"))
    # table panel
    ty = yb + 74
    cols = [20, 230, 400, 560]
    heads = ["Interface", "Unnecessary escalations", "Wrong ALLOW on controls", "Generation agreement"]
    b.append(T(20, ty - 10, f"Controls (n = {A['norm_total']['A']['control_n']}) and scoring validity, total / first-token", "t b"))
    for x, h in zip(cols, heads, strict=True):
        b.append(T(x, ty + 14, h, "s b"))
    b.append(line(20, ty + 22, W - 20, ty + 22, "sep"))
    for i, (c, name) in enumerate(conds):
        y = ty + 42 + i * 22
        t_, f_ = A["norm_total"][c], A["norm_first"][c]
        ag = fg[c]["agree_by_norm"]
        b += [T(cols[0], y, f"{c} · {name}", "t"),
              T(cols[1], y, f'{t_["control_unnecessary_escalations"]} / {f_["control_unnecessary_escalations"]}', "t"),
              T(cols[2], y, f'{t_["wrong_allow"]["controls"]} / {f_["wrong_allow"]["controls"]}', "t"),
              T(cols[3], y, f'{ag["total"]}/{fg[c]["n"]} / {ag["first"]}/{fg[c]["n"]}', "t")]
    yc = ty + 42 + 4 * 22 + 12
    dt, df = A["norm_total"]["D"]["conflict_escalations"], A["norm_first"]["D"]["conflict_escalations"]
    b += [T(20, yc, f"Mapping D's total-score result ({dt}/{n}) fails the agreement check: the model answers “X = DECISION”, which the scored", "s"),
          T(20, yc + 18, f"continuation “X\\n” penalises. It is not a verified improvement. The first-token figure ({df}/{n}) is the better-supported", "s"),
          T(20, yc + 36, f"sensitivity result, not a validated replacement. Gap: {100 * (dt - df) / n:.1f} percentage points. "
            "Free-generation checks cover 12 items per interface.", "s"),
          T(20, yc + 54, "B's first-token scoring also disagreed with generation on 2 of 12 items; it gives 0 escalations under either scoring.", "s"),
          T(20, yc + 72, "Source: experiments/mechinterp/results/label_diag/analysis.json and freegen_by_norm.json", "s mut")]
    return doc(W, yc + 90, "Correct escalations by answer interface and scoring",
               "; ".join(f"{c}: total {A['norm_total'][c]['conflict_escalations']}, first-token {A['norm_first'][c]['conflict_escalations']}"
                         for c, _ in conds), b)


# ----------------------------------------------------------------------------------------------------------- diagrams


def fig_pipeline(d) -> str:
    W, H = 900, 712
    b = [T(20, 32, "Research pipeline", "title"),
         T(20, 52, "Offline evaluation of a local model on synthetic items. No production service; no tool or external action is executed.", "sub")]
    b += box(20, 72, 250, 96, ["Synthetic scenarios", f"{d['n_bases']} base scenarios → {d['n_items']} items",
                                 f"{d['n_direct']} direct (primary) · {d['n_dual']} dual (stress)", f"generate.py, seed {d['seed']}"])
    b += box(300, 72, 250, 96, ["Formal oracle", "three-valued (Kleene) evaluation", "status TRUE / FALSE / UNKNOWN /",
                                  "INCONSISTENT → expected decision"])
    b += arrow([(270, 120), (298, 120)])
    b += box(20, 206, 530, 76, ["Qwen3.5-9B · bf16 · Hugging Face transformers · Apple MPS (local)",
                                  "official chat template; forced-choice scoring of ALLOW / DENY / REQUEST_EVIDENCE / ESCALATE",
                                  "forward hooks record the residual stream (read-only)"], "boxa")
    b += arrow([(145, 168), (145, 204)], "items", 152, 190)
    b += box(20, 320, 250, 76, ["Behavioural labels", f"decision, margin, confidence ({d['n_items']})",
                                  "error = decision ≠ oracle decision"])
    b += box(290, 320, 260, 76, ["Activations", f"blocks {', '.join(map(str, d['blocks']))} × 2 positions × {d['hidden']}",
                                   "float16 (not distributed)"])
    b += arrow([(145, 282), (145, 318)])
    b += arrow([(420, 282), (420, 318)])
    b += box(20, 436, 530, 124, ["Linear probes on stored activations (offline, CPU)", "Task A · decode oracle status (4 classes)",
                                  "Task B · predict the model's errors", f"Scenario-disjoint: leave-one-scenario-out over {d['n_bases']} scenarios;",
                                  "fold-local preprocessing; shuffled-label controls; cluster bootstrap",
                                  "Baselines: output margin, TF-IDF, text + margin, construction-aware features"], "boxc", lead=17)
    b += arrow([(145, 396), (145, 434)], "Task B labels, margin", 152, 420)
    b += arrow([(420, 396), (420, 434)])
    b += arrow([(550, 120), (572, 120), (572, 416), (500, 416), (500, 434)])
    b += [T(580, 300, "oracle", "s"), T(580, 316, "labels", "s")]
    b += box(630, 206, 250, 190, ["Answer-interface diagnostic", "separate behavioural evaluation",
                                    f"{d['n_diag']} items: {d['n_diag_conflict']} CONFLICT + {d['n_diag'] - d['n_diag_conflict']} controls",
                                    "4 label interfaces (A–D)", "3 scoring normalisations",
                                    "free-generation agreement check", "no activations · no intervention"], "boxb", lead=21)
    b += arrow([(550, 252), (628, 252)]) + [T(580, 226, "same", "s"), T(580, 242, "model", "s")]
    b += box(20, 600, 860, 64, ["Preserved raw outputs → SHA-256 manifest → verify_integrity.py",
                                  "standard library only; recomputes headline counts from raw log-probs without the model"])
    b += arrow([(285, 560), (285, 598)])
    b += arrow([(755, 396), (755, 598)])
    b.append(T(20, 694, "Arrows show data flow. Probes read stored activations after the forward passes and do not feed back into the model.", "s mut"))
    return doc(W, H, "Research pipeline", "Data flow from synthetic scenarios and oracle through local model scoring, activation capture, "
               "scenario-disjoint probes, a separate answer-interface diagnostic and checksum-based integrity verification.", b)


def fig_activation(d) -> str:
    W, H = 900, 750
    lt, blocks = d["layer_types"], d["blocks"]
    n_full = sum(t == "full_attention" for t in lt)
    b = [T(20, 32, "Activation extraction and probing", "title"),
         T(20, 52, f"Qwen3.5-9B text decoder: {d['n_layers']} blocks, hidden size {d['hidden']}; hybrid attention. "
           "Sampled blocks are read with forward hooks.", "sub")]
    b += box(20, 72, 330, 84, ["Input", "system prompt: decision rules and the four labels",
                                 "user message: policy and evidence lines"])
    b += box(390, 72, 300, 84, ["Official chat template + tokenizer", "enable_thinking = False",
                                  "prompt ends: assistant, empty <think> block"])
    b += arrow([(350, 114), (388, 114)])
    # decoder strip
    y0, cw, gap, x0 = 210, 15, 3, 40
    b.append(f'<rect x="20" y="{y0 - 22}" width="860" height="132" rx="6" class="box"/>')
    b.append(T(32, y0 - 4, f"Decoder blocks 0–{len(lt) - 1}", "t b"))
    b += arrow([(540, 156), (540, y0 - 24)])
    for i, t in enumerate(lt):
        x = x0 + i * (cw + gap)
        b.append(f'<rect x="{x}" y="{y0 + 8}" width="{cw}" height="30" rx="2" class="{"cellf" if t == "full_attention" else "celll"}"/>')
        if i in blocks:
            b.append(f'<path class="head" d="M{x + cw / 2 - 6},{y0 + 54} L{x + cw / 2 + 6},{y0 + 54} L{x + cw / 2},{y0 + 44} z"/>')
            b.append(T(x + cw / 2, y0 + 70, str(i) + ("*" if i == blocks[-1] else ""), "s b", "middle"))
    xe = x0 + len(lt) * (cw + gap)
    b += box(xe + 10, y0 + 4, 120, 38, ["Final RMSNorm"], size=12)
    b += box(xe + 144, y0 + 4, 90, 38, ["LM head"], size=12)
    b += arrow([(xe - 2, y0 + 23), (xe + 8, y0 + 23)]) + arrow([(xe + 130, y0 + 23), (xe + 142, y0 + 23)])
    lg = y0 + 88
    b += [f'<rect x="40" y="{lg}" width="14" height="10" class="cellf"/>', T(60, lg + 9, f"full attention ({n_full}, every 4th block)", "s"),
          f'<rect x="270" y="{lg}" width="14" height="10" class="celll"/>', T(290, lg + 9, f"Gated DeltaNet linear attention ({len(lt) - n_full})", "s"),
          f'<path class="head" d="M520,{lg + 10} L532,{lg + 10} L526,{lg} z"/>', T(540, lg + 9, "sampled block", "s"),
          T(650, lg + 9, f"* block {blocks[-1]}: read before final RMSNorm", "s")]
    b += box(20, 360, 640, 66, ["Forward hooks (read-only; logits bitwise unchanged)",
                                  f"blocks {', '.join(map(str, blocks[:-1]))}: block output = hidden_states[b+1];  "
                                  f"block {blocks[-1]}: pre-norm residual stream"], "boxa")
    b += arrow([(340, 320), (340, 358)])
    b += box(20, 464, 310, 66, ["Last prompt token (primary)", f"pre-decision position · {d['hidden']}-d"], "boxa")
    b += box(350, 464, 310, 66, ["Mean over user-message tokens", f"secondary · {d['hidden']}-d"], "boxa")
    b += arrow([(175, 426), (175, 462)]) + arrow([(505, 426), (505, 462)])
    b += box(20, 574, 640, 106, ["Offline analysis on stored activations (CPU)",
                                   f"float16 array: {d['n_items']} items × {len(blocks)} blocks × 2 positions × {d['hidden']}",
                                   "standardised, L2 logistic regression; C chosen by inner grouped CV",
                                   f"outer leave-one-scenario-out over {d['n_bases']} scenarios → Task A (status), Task B (errors)"], "boxc")
    b += arrow([(175, 530), (175, 572)]) + arrow([(505, 530), (505, 572)])
    b += box(690, 360, 190, 170, ["Four-label scoring", "teacher-forced log-prob", "of “LABEL\\n” for each of", "ALLOW, DENY,",
                                    "REQUEST_EVIDENCE,", "ESCALATE; decision =", "argmax, margin = top − 2nd"], "boxb", lead=19)
    b += [line(xe + 189, y0 + 42, xe + 189, y0 + 78, "arrow")] + arrow([(xe + 189, y0 + 110), (xe + 189, 358)])
    b += arrow([(785, 530), (785, 628), (662, 628)], "Task B labels", 690, 620)
    b.append(T(20, 720, "Probes are fitted after the forward passes on saved arrays; they are not part of the model and do not influence its decisions.", "s mut"))
    return doc(W, H, "Activation extraction", f"Prompt rendering, the {d['n_layers']}-block hybrid decoder with sampled full-attention blocks "
               f"{', '.join(map(str, blocks))}, last-token and user-mean representations of size {d['hidden']}, "
               "offline scenario-disjoint linear probes, and separate four-label behavioural scoring.", b)


# ----------------------------------------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true", help="also export PNG at 2x via rsvg-convert, if installed")
    args = ap.parse_args()
    d = data()
    OUT.mkdir(exist_ok=True)
    figs = {"research_pipeline.svg": fig_pipeline, "activation_extraction.svg": fig_activation,
            "conflict_decisions.svg": fig_conflict, "layerwise_decoding.svg": fig_layerwise,
            "error_prediction.svg": fig_error, "interface_sensitivity.svg": fig_interface}
    for name, fn in figs.items():
        (OUT / name).write_text(fn(d))
        print("wrote", f"figures/{name}")
    if args.png:
        tool = shutil.which("rsvg-convert")
        if not tool:
            print("rsvg-convert not found; PNG export skipped (SVGs are the canonical output)")
            return
        for name in figs:
            subprocess.run([tool, "-z", "2", "-o", str(OUT / name.replace(".svg", ".png")), str(OUT / name)], check=True)
            print("wrote", f"figures/{name.replace('.svg', '.png')}")


if __name__ == "__main__":
    main()
