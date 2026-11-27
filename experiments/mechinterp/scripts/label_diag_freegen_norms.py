"""Descriptive (post-hoc, D46): forced-choice vs free-generation agreement on the 12 pre-declared free-generation
items, per condition, for each scoring normalisation (total / first-token / per-token mean), plus the
generation's first-line format. Explains termination-handling effects; changes no primary result.
Usage: uv run python scripts/label_diag_freegen_norms.py > results/label_diag/freegen_by_norm.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_diag_analyze as A


def main():
    _, rec = A.load()
    fg = [json.loads(line) for line in (A.OUT / "freegen.jsonl").open()][1:]
    out = {"label": "POST-HOC descriptive (D46)"}
    for c in "ABCD":
        rows = [g for g in fg if g["condition_id"] == c]
        res = {}
        for norm in A.NORMS:
            res[norm] = sum(max(rec[(c, g["item_id"])][norm], key=rec[(c, g["item_id"])][norm].get) == g["parsed_decision"] for g in rows)
        first_lines = [next(line for line in g["generated"].splitlines() if line.strip()) for g in rows]
        fmt = {"bare_candidate_line": sum(bool(re.fullmatch(r"\**\s*[A-Z_]+\s*\**", fl.strip())) for fl in first_lines),
               "candidate_equals_name": sum(" = " in fl for fl in first_lines)}
        out[c] = {"n": len(rows), "agree_by_norm": res, "first_line_format": fmt}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
