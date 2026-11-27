"""Technical smoke test (<=12 existing exploratory items; reports/43 §5). Not a scientific evaluation.

Checks: load, MPS operation / CPU-fallback warnings, peak memory, hook locations and shapes, hooks do not
modify logits, hook capture == output_hidden_states, label tokenisation, forced-choice scoring, greedy
generation agreement (12 items), run-to-run reproducibility, forward-pass throughput.
Usage: HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
from mechinterp import model_io as M

ITEMS = ROOT / "data/synthetic/exploratory_v1/items.jsonl"
SYSTEM = (ROOT / "data/synthetic/exploratory_v1/system_prompt.txt").read_text()


def mem():
    return {"mps_current_gb": round(torch.mps.current_allocated_memory() / 1e9, 2),
                "mps_driver_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2),
                "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)}


def main():
    out = {"label": "TECHNICAL SMOKE TEST (not scientific)", "env": {
        "torch": torch.__version__, "mps": torch.backends.mps.is_available(),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}}
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        tok, model, load_s = M.load()
        out["load_seconds"] = round(load_s, 1)
        out["mem_after_load"] = mem()
        layers, path = M.decoder_layers(model)
        out["decoder_path"] = path
        out["n_decoder_layers"] = len(layers)
        out["layer_types_at_blocks"] = {b: type(layers[b].self_attn if hasattr(layers[b], "self_attn") else layers[b]).__name__
                                        for b in M.BLOCKS}
        out["label_tokenisation"] = {L: tok(f"{L}\n", add_special_tokens=False)["input_ids"] for L in M.LABELS}
        out["label_first_tokens_distinct"] = len({v[0] for v in out["label_tokenisation"].values()}) == 4

        items = [json.loads(line) for line in ITEMS.open()]
        sel = [i for i in items if i["base_id"] == "DA01" and i["family"] in ("existential", "universal")
               and i["condition"] in ("POS", "NEG", "MISS")]
        assert len(sel) == 12, len(sel)
        rows = []
        for k, it in enumerate(sel):
            t0 = time.time()
            r = M.score_item(tok, model, SYSTEM, it["user_prompt"])
            r2 = M.score_item(tok, model, SYSTEM, it["user_prompt"], capture=False) if k < 4 else None
            gen = M.generate(tok, model, SYSTEM, it["user_prompt"])
            parsed, pstat = M.parse_decision(gen)
            row = {"item_id": it["item_id"], "family": it["family"], "condition": it["condition"], "wording": it["wording"],
                       "gold": it["oracle_decision"], "n_prompt_tokens": r["n_prompt_tokens"], "decision": r["decision"],
                       "margin": round(r["margin"], 3), "first_token_decision": r["first_token_decision"],
                       "label_logprob": {k2: round(v, 3) for k2, v in r["label_logprob"].items()},
                       "generated": gen[:200], "generated_decision": parsed, "parse_status": pstat,
                       "forward_seconds": [round(x, 2) for x in r["forward_seconds"]], "item_seconds": round(time.time() - t0, 1)}
            if r2 is not None:
                row["repeat_identical_label_logprobs"] = r["label_logprob"] == r2["label_logprob"]
                row["repeat_max_abs_logprob_diff"] = max(abs(r["label_logprob"][L] - r2["label_logprob"][L]) for L in M.LABELS)
            if k == 0:
                # hooks vs no hooks; hook capture vs output_hidden_states
                prompt = M.render(tok, SYSTEM, it["user_prompt"])
                ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]], device="mps")
                store: dict = {}
                hs = M.attach_hooks(model, store)
                with torch.no_grad():
                    o_h = model(input_ids=ids, output_hidden_states=True)
                for h in hs:
                    h.remove()
                with torch.no_grad():
                    o_n = model(input_ids=ids)
                hid = o_h.hidden_states
                row["hook_checks"] = {
                    "n_hidden_states": len(hid),
                    "shapes": {b: list(store[b].shape) for b in M.BLOCKS},
                    "logits_bitwise_equal_hooks_vs_none": bool(torch.equal(o_h.logits, o_n.logits)),
                    "hook_equals_hidden_states[b+1]": {b: bool(torch.equal(store[b], hid[b + 1])) for b in M.BLOCKS},
                    "hook_max_abs_diff_vs_hidden_states[b+1]": {b: float((store[b].float() - hid[b + 1].float()).abs().max()) for b in M.BLOCKS},
                    "rep_dims": {b: list(r["reps"][b]["last"].shape) for b in M.BLOCKS},
                }
            rows.append(row)
            print(k, row["decision"], row["generated_decision"], row["gold"], row["item_seconds"], flush=True)
        out["mem_peak"] = mem()
        out["rows"] = rows
        fw = [x for r in rows for x in r["forward_seconds"]]
        out["throughput"] = {"mean_forward_s": round(sum(fw) / len(fw), 2), "max_forward_s": max(fw),
                                 "mean_item_s_incl_generation": round(sum(r["item_seconds"] for r in rows) / len(rows), 1)}
        out["agreement_forced_vs_generated"] = f"{sum(r['decision'] == r['generated_decision'] for r in rows)}/{len(rows)}"
        out["agreement_full_vs_first_token"] = f"{sum(r['decision'] == r['first_token_decision'] for r in rows)}/{len(rows)}"
    out["warnings"] = sorted({str(w.message)[:200] for w in wl})
    dest = HERE / "results/smoke_test.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    print("WROTE", dest)


if __name__ == "__main__":
    main()
