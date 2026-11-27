"""Model loading, prompt rendering, forced-choice label scoring and residual-stream capture for the
mechanistic-interpretability pilot (Qwen/Qwen3.5-9B, bf16, MPS). Native PyTorch hooks only.

Pre-declared (reports/43 §5):
- prompt = chat template(system = pilot SYSTEM_PROMPT, user = frozen item prompt), add_generation_prompt=True,
  enable_thinking=False  -> ends with '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'
- forced choice: total teacher-forced log-prob of each continuation f"{label}\\n" appended at the token-ID level
  (prompt IDs + continuation IDs; the prompt prefix is identical across labels)
- capture: residual stream (decoder-layer output) of blocks 7, 15, 23, 31 (== hidden_states[8,16,24,32]),
  at the LAST PROMPT TOKEN (pre-decision) and the mean over user-message tokens.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import torch

LABELS = ("ALLOW", "DENY", "REQUEST_EVIDENCE", "ESCALATE")
MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BLOCKS = (7, 15, 23, 31)  # 0-indexed decoder blocks; outputs == hidden_states[b + 1]


def load(device: str = "mps"):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16)
    model.eval().to(device)
    return tok, model, time.time() - t0


def decoder_layers(model):
    for path in ("model.language_model.layers", "model.layers", "language_model.model.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj, path
        except AttributeError:
            continue
    raise RuntimeError("decoder layers not found")


def render(tok, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)


def user_span(tok, prompt_ids: list[int], user: str) -> tuple[int, int]:
    """Token span [a, b) of the user message content within prompt_ids (located by decoding offsets)."""
    text = tok.decode(prompt_ids)
    start = text.find(user)
    if start < 0:
        raise RuntimeError("user text not found in rendered prompt")
    end = start + len(user)
    # map char offsets -> token indices by cumulative decoding
    _pos, a, b = 0, None, None
    for i in range(len(prompt_ids)):
        piece = tok.decode(prompt_ids[: i + 1])
        if a is None and len(piece) > start:
            a = i
        if len(piece) >= end:
            b = i + 1
            break
        len(piece)
    return a, b


@dataclass
class Capture:
    acts: dict
    handles: list


def attach_hooks(model, store: dict):
    layers, _ = decoder_layers(model)
    handles = []
    for b in BLOCKS:
        def hook(_m, _inp, out, b=b):
            h = out[0] if isinstance(out, tuple) else out
            store[b] = h.detach()
        handles.append(layers[b].register_forward_hook(hook))
    return handles


@torch.no_grad()
def score_item(tok, model, system: str, user: str, device: str = "mps", capture: bool = True):
    """Returns dict with label log-probs (full continuation and first token), decision, margin and
    (optionally) captured representations at the last prompt token and mean over user tokens."""
    prompt = render(tok, system, user)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a, b = user_span(tok, p_ids, user)
    out = {"n_prompt_tokens": len(p_ids), "user_span": [a, b], "label_tokens": {}, "label_logprob": {}}
    store: dict = {}
    handles = attach_hooks(model, store) if capture else []
    reps = None
    first_tok_lp = None
    timings = []
    try:
        for li, lab in enumerate(LABELS):
            c_ids = tok(f"{lab}\n", add_special_tokens=False)["input_ids"]
            out["label_tokens"][lab] = c_ids
            ids = torch.tensor([p_ids + c_ids], device=device)
            t0 = time.time()
            logits = model(input_ids=ids).logits.float()
            torch.mps.synchronize() if device == "mps" else None
            timings.append(time.time() - t0)
            logp = torch.log_softmax(logits[0], dim=-1)
            n = len(p_ids)
            lp = sum(float(logp[n - 1 + j, c_ids[j]]) for j in range(len(c_ids)))
            out["label_logprob"][lab] = lp
            if li == 0:
                first_tok_lp = {L: float(logp[n - 1, tok(f"{L}\n", add_special_tokens=False)["input_ids"][0]]) for L in LABELS}
                out["next_token_logits_last_prompt"] = logits[0, n - 1].cpu()
                if capture:
                    reps = {blk: {"last": store[blk][0, n - 1].float().cpu(),
                                  "user_mean": store[blk][0, a:b].float().mean(0).cpu()} for blk in BLOCKS}
    finally:
        for h in handles:
            h.remove()
    ranked = sorted(out["label_logprob"].items(), key=lambda kv: -kv[1])
    out["decision"] = ranked[0][0]
    out["margin"] = ranked[0][1] - ranked[1][1]
    out["first_token_logprob"] = first_tok_lp
    out["first_token_decision"] = max(first_tok_lp, key=first_tok_lp.get)
    out["forward_seconds"] = timings
    out["reps"] = reps
    return out


# Parser copied verbatim in logic from src/authorization_closure/model_client.parse_decision (pilot amendment A1)
def _clean(line: str) -> str:
    s = line.strip().strip("*`#>\"' ").strip()
    s = re.sub(r"^(decision|answer|verdict)\s*[:\-]\s*", "", s, flags=re.IGNORECASE)
    s = s.strip("*`\"' ").rstrip(".:!").strip()
    return s.upper()


def parse_decision(text: str):
    if not text or not text.strip():
        return None, "EMPTY"
    first = next(line for line in text.splitlines() if line.strip())
    c = _clean(first)
    if c in LABELS:
        return c, "STRICT"
    m = re.match(r"^(ALLOW|DENY|REQUEST_EVIDENCE|ESCALATE)\s*[:\-–—.,]\s*\S", c)
    if m:
        return m.group(1), "STRICT"
    found = {m.group(0).upper().replace(" ", "_") for m in
             re.finditer(r"\b(ALLOW|DENY|REQUEST[_ ]EVIDENCE|ESCALATE)\b", text, flags=re.IGNORECASE)}
    if len(found) == 1:
        return found.pop(), "LENIENT"
    return None, "AMBIGUOUS" if found else "INVALID"


@torch.no_grad()
def generate(tok, model, system: str, user: str, max_new_tokens: int = 80, device: str = "mps") -> str:
    prompt = render(tok, system, user)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
