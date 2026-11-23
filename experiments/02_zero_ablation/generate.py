"""Experiment 02: is the evil response_avg direction NECESSARY?  All-layer zero ablation.  Env: interp.

    python experiments/02_zero_ablation/generate.py hooks  [--limit N --max_new T]   conditions 1, 2, 4, 5
    python experiments/02_zero_ablation/generate.py ortho  [--limit N --max_new T]   condition 3 + equivalence check
(two processes: only one copy of the 7B model fits in 24 GB, and ortho edits the loaded weights in place;
nothing on disk is changed.)

Direction: v = evil_response_avg_diff[20] (the vector whose steering gave evil 96.1 in task 4),
v_hat = v / ||v||  [3584], fp32. The SAME v_hat is used at every layer and every position.

Zero ablation, two implementations that must agree (level 1, the stream is a plain sum of writes):
  hooks : every writer's output o [batch, pos, 3584] -> o - (o @ v_hat)[..., None] * v_hat
          writers = embed_tokens, every self_attn.o_proj, every mlp.down_proj  (1 + 28 + 28 = 57)
          => the stream has no v_hat component at any layer or position, so no reader ever sees one.
  ortho : P = I - v_hat v_hat^T  [3584, 3584]
          o_proj.weight [3584, 3584] and down_proj.weight [3584, 18944]  ([out, in], output = W x):  W <- P W
          embed_tokens.weight [vocab, 3584] (one token per ROW):                                      E <- E P
          o_proj / down_proj have no bias in Qwen2, lm_head is not tied to the embedding (checked below).
Conditions (greedy decoding, max 1000 new tokens like the paper's eval):
  1 evil_base        5 evil system prompts x 20 eval questions, no intervention   (known answer: high evil)
  2 evil_ablate      same prompts, hooks
  3 evil_ortho       same prompts, orthogonalized weights                          (should match 2)
  4 noprompt_base    20 questions, Qwen's default system prompt, no intervention   (task 4: evil 0.0)
  5 noprompt_ablate  same, hooks                                                   (does zero ablation push it toward evil?)
For every answer we also run one forward pass over prompt + answer and record the "dial" v_hat^T h_l at
hidden_states[l], l = 0..27 (raw stream; 28 is post-norm and skipped), averaged over prompt tokens
(position 0 left out: attention-sink token with a huge norm) and over answer tokens. For conditions 2/3/5
the dial must be ~0 at every layer (check); conditions 1/4 give the natural values (mu_l for later).
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import torch

REPO = "/data/coe9/unpersona"
sys.path.insert(0, REPO)
from unpersona.model import load_model, N_LAYERS, D_MODEL                          # noqa: E402
from unpersona.prompts import build_prompt, load_questions                          # noqa: E402
from unpersona.vectors import load_vectors                                          # noqa: E402

TRAIT, VARIANT, LAYER = "evil", "response_avg", 20
JUDGED = "/data/coe9/artifacts/unpersona/judged_final/Qwen2.5-7B-Instruct"
OUT = "/data/coe9/artifacts/unpersona/exp02_zero_ablation"
torch.manual_seed(0)


def evil_systems():
    """The 5 evil system prompts exactly as the extraction run used them (read from its prompts)."""
    p = pd.read_csv(f"{JUDGED}/{TRAIT}_pos_instruct.csv")
    s = sorted(set(re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", x, flags=re.S).group(1) for x in p.prompt))
    assert len(s) == 5, len(s)
    return s


def rows_for(tok, cond, systems, qs, limit):
    if cond.startswith("evil"):
        rows = [dict(sys_idx=i, q_idx=j, system=s, question=q, prompt=build_prompt(tok, q, s))
                for i, s in enumerate(systems) for j, q in enumerate(qs)]
    else:  # system=None -> the template inserts Qwen's default system prompt (trap, see prompts.py)
        rows = [dict(sys_idx=-1, q_idx=j, system="<qwen default>", question=q, prompt=build_prompt(tok, q, None))
                for j, q in enumerate(qs)]
    return rows[:limit] if limit else rows


class ZeroAblate:
    """Forward hooks on the 57 writers; each removes the v_hat component of its output (computed in fp32)."""

    def __init__(self, hf, vhat):
        self.mods = [hf.model.embed_tokens] + [m for L in hf.model.layers for m in (L.self_attn.o_proj, L.mlp.down_proj)]
        self.vhat, self.handles = vhat, []

    def hook(self, mod, inp, out):                       # out [batch, pos, 3584] bf16
        o = out.float()
        o = o - (o @ self.vhat)[..., None] * self.vhat   # (o @ v_hat) [batch, pos]: the dial of this write
        return o.to(out.dtype)

    def __enter__(self):
        self.handles = [m.register_forward_hook(self.hook) for m in self.mods]
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def orthogonalize(hf, vhat):
    """W <- P W for o_proj / down_proj (P acts on the output side), E <- E P for the embedding rows."""
    assert not hf.config.tie_word_embeddings, "embedding tied to lm_head: E <- E P would also change the readout"
    v = vhat.to(hf.device)                                                   # [3584] fp32
    for L in hf.model.layers:
        for lin in (L.self_attn.o_proj, L.mlp.down_proj):
            assert lin.bias is None
            W = lin.weight.float()                                           # [3584, in]
            lin.weight.copy_((W - torch.outer(v, v @ W)).to(lin.weight.dtype))  # v (v^T W): [3584, in]
    E = hf.model.embed_tokens.weight                                         # [vocab, 3584]
    for a in range(0, E.shape[0], 16384):                                    # chunks: fp32 copy of all rows = 2.2 GB
        e = E[a:a + 16384].float()
        E[a:a + 16384] = (e - (e @ v)[:, None] * v[None, :]).to(E.dtype)    # (e @ v) [chunk]: dial of each token row
    # check: no writer has any v_hat component left (up to bf16 rounding)
    worst = max((v @ L.mlp.down_proj.weight.float()).abs().max().item() for L in hf.model.layers)
    print(f"ortho: max |v_hat^T W_down| after = {worst:.2e}; max |E v_hat| = {(E.float() @ v).abs().max().item():.2e}")


@torch.no_grad()
def generate(hf, tok, prompts, max_new, bs=20):
    tok.padding_side = "left"                        # last prompt token at position -1 for every row
    eos = hf.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    out = []
    for a in range(0, len(prompts), bs):
        enc = tok(prompts[a:a + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(hf.device)
        g = hf.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        new = g[:, enc.input_ids.shape[1]:]          # [bs, n_new]
        for r in new:
            ids = r.tolist()
            fin = any(t in eos for t in ids)
            out.append(dict(answer=tok.decode(r, skip_special_tokens=True), finished=fin))
        print(f"   generated {min(a + bs, len(prompts))}/{len(prompts)}", flush=True)
    return out


@torch.no_grad()
def dial_and_logits(hf, tok, prompt, answer, vhat):
    """One forward pass over prompt + answer (batch 1, no padding).
    returns dial_prompt [28], dial_answer [28] (mean of v_hat^T hidden_states[l] over tokens, l = 0..27, fp32)
    and logits [vocab] fp32 at the last prompt token (the distribution of the first answer token)."""
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(answer, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + a_ids], device=hf.device)                    # [1, T]
    hs = hf.model(ids, output_hidden_states=True).hidden_states              # 29 x [1, T, 3584]
    H = torch.stack(hs[:N_LAYERS])[:, 0].float()                             # [28, T, 3584]
    d = H @ vhat                                                             # [28, T]  the dial per layer, token
    P = len(p_ids)
    dial_p = d[:, 1:P].mean(1)                                               # skip position 0 (sink)
    dial_a = d[:, P:].mean(1) if len(a_ids) else torch.full((N_LAYERS,), float("nan"), device=d.device)
    logits = hf.lm_head(hs[N_LAYERS][0, P - 1]).float()                      # hs[28] is already final-normed
    return dial_p.cpu(), dial_a.cpu(), logits.cpu()


def run_condition(hf, tok, cond, rows, vhat, max_new, ablate):
    t0 = time.time()
    print(f"=== {cond}: {len(rows)} prompts, ablation={ablate}", flush=True)
    ctx = ZeroAblate(hf, vhat) if ablate else torch.no_grad()
    with ctx:
        gen = generate(hf, tok, [r["prompt"] for r in rows], max_new)
        D_p, D_a, LG = [], [], []
        for r, g in zip(rows, gen):
            dp, da, lg = dial_and_logits(hf, tok, r["prompt"], g["answer"], vhat)
            D_p.append(dp); D_a.append(da); LG.append(lg)
    df = pd.DataFrame([{**r, **g} for r, g in zip(rows, gen)])
    df["n_answer_tokens"] = [len(tok(a, add_special_tokens=False).input_ids) for a in df.answer]
    df.drop(columns="prompt").to_csv(f"{OUT}/{cond}.csv", index=False)
    D_p, D_a = torch.stack(D_p), torch.stack(D_a)                            # [n, 28] each
    torch.save(dict(dial_prompt=D_p, dial_answer=D_a, logits=torch.stack(LG)), f"{OUT}/{cond}_internals.pt")
    print(f"   dial at layer 20: prompt tokens {D_p[:, 20].mean():+.2f}, answer tokens {D_a[:, 20].nanmean():+.2f}; "
          f"max |dial| over layers 0-27 {max(D_p.abs().max(), D_a.nan_to_num().abs().max()):.3f}")
    print(f"   finished (EOS) {df.finished.mean():.0%}, mean answer tokens {df.n_answer_tokens.mean():.0f}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hooks", "ortho"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N prompts per condition")
    ap.add_argument("--max_new", type=int, default=1000)
    a = ap.parse_args()
    global OUT
    if a.limit:
        OUT += "/smoke"
    os.makedirs(OUT, exist_ok=True)

    model = load_model()
    hf, tok = model._module, model.tokenizer                                 # plain HF model: hooks + generate
    gc = hf.generation_config
    print(f"generation_config: repetition_penalty={gc.repetition_penalty} (kept; do_sample=False so temperature/top_p/top_k unused)")
    vhat = load_vectors(TRAIT, VARIANT)[LAYER]
    vhat = (vhat / vhat.norm()).to(hf.device)                                # [3584] fp32, unit length
    systems, qs = evil_systems(), load_questions(TRAIT, "eval")
    meta = dict(model="Qwen/Qwen2.5-7B-Instruct", dtype="bf16", attn="eager", trait=TRAIT, variant=VARIANT,
                layer=LAYER, decoding="greedy", max_new=a.max_new, repetition_penalty=gc.repetition_penalty,
                torch=torch.__version__, transformers=__import__("transformers").__version__)
    json.dump(meta, open(f"{OUT}/meta_{a.mode}.json", "w"), indent=1)

    if a.mode == "hooks":
        for cond, ablate in [("evil_base", False), ("evil_ablate", True), ("noprompt_base", False), ("noprompt_ablate", True)]:
            run_condition(hf, tok, cond, rows_for(tok, cond, systems, qs, a.limit), vhat, a.max_new, ablate)
    else:
        orthogonalize(hf, vhat)
        df = run_condition(hf, tok, "evil_ortho", rows_for(tok, "evil_ortho", systems, qs, a.limit), vhat, a.max_new, False)
        # equivalence check against the hooks run (same prompts, same order)
        ref = torch.load(f"{OUT}/evil_ablate_internals.pt")["logits"]       # [n, vocab]
        base = torch.load(f"{OUT}/evil_base_internals.pt")["logits"]
        new = torch.load(f"{OUT}/evil_ortho_internals.pt")["logits"]
        kl = lambda p, q: (p.log_softmax(-1).exp() * (p.log_softmax(-1) - q.log_softmax(-1))).sum(-1)  # KL(p||q) [n]
        h = pd.read_csv(f"{OUT}/evil_ablate.csv")
        same = (h.answer.fillna("") == df.answer.fillna("")).mean()
        res = dict(kl_hooks_vs_ortho=kl(ref, new).mean().item(), kl_base_vs_hooks=kl(base, ref).mean().item(),
                   top1_agree=(ref.argmax(-1) == new.argmax(-1)).float().mean().item(), identical_answers=float(same))
        json.dump(res, open(f"{OUT}/ortho_check.json", "w"), indent=1)
        print("ortho check (first answer token at the last prompt position):", json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
