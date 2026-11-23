"""Experiment 02b: all-layer MEAN ablation along v̂_20 (the standard evil direction).  Env: interp.

    python experiments/02_zero_ablation/mean_ablation.py [--limit N --max_new T]

v̂_20 (the standard evil direction) = evil_response_avg_diff[20] / norm  [3584] fp32, the same fixed vector at
every layer, as in the zero-ablation run (generate.py). The only change from that run: the reading along
v̂_20 is set to its NORMAL value instead of 0.

Step 1  normal values. Forward passes over the saved no-prompt answers (noprompt_base.csv: Qwen's default
        system prompt, no ablation). At every stream point l = 0..28 (l < 28: hidden_states[l], the input of
        block l; l = 28: the input of the final norm = block 27's output, NOT HF's post-norm hidden_states[28]):
            reading = v̂_20ᵀh_l                              one number per token
            mu_prompt[l] = mean over prompt tokens  (all 20 runs pooled, first token left out)
            mu_answer[l] = mean over answer tokens  (all 20 runs pooled)
Step 2  mean ablation at every layer and every token:
            h' = h - (v̂_20ᵀh - mu_l) v̂_20             reading becomes exactly mu_l; other 3583 directions untouched
        prompt positions get mu_prompt, generated positions get mu_answer.
        Implementation through the writers (the stream is a plain sum of writes, level 1):
            embed_tokens out : remove v̂_20 part, add mu[0] v̂_20                -> reading at block 0 input = mu[0]
            o_proj out (l)   : remove v̂_20 part                                 -> mid-block reading stays mu[l]
            down_proj out (l): remove v̂_20 part, add (mu[l+1] - mu[l]) v̂_20     -> block l output = mu[l+1]
        So the mid-block point (after attention, before the MLP) is held at mu[l], the block-input value.
        The FIRST real token (<|im_start|>, the attention sink) is left untouched: it sees only itself, so its
        stream is identical in every run, i.e. already "normal". Left-padding positions are left untouched.
Conditions (greedy, same prompts and settings as generate.py):
    evil_mean       5 evil system prompts x 20 eval questions     the necessity test
    noprompt_mean   Qwen default x 20 questions                   known answer: should stay ~0 evil
Check: recorded readings along v̂_20 must equal mu_prompt / mu_answer at every layer.
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

REPO = "/data/coe9/unpersona"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unpersona.model import load_model, N_LAYERS                                   # noqa: E402
from unpersona.prompts import build_prompt, load_questions                          # noqa: E402
from unpersona.vectors import load_vectors                                          # noqa: E402
from generate import evil_systems, rows_for, TRAIT, VARIANT, LAYER                  # noqa: E402

BASE = "/data/coe9/artifacts/unpersona/exp02_zero_ablation"
torch.manual_seed(0)


class MeanAblate:
    """Forward hooks on the 57 writers. State set before each call:
         split      positions >= split are answer tokens (prefill of a prompt: split = T, all prompt)
         first_idx  [batch] index of each row's first real token (after left padding); positions <= it untouched
       A forward pass with one new token (T == 1) is a generation step: answer token, never the first token."""

    def __init__(self, hf, vhat, mu_p, mu_a):
        self.emb = hf.model.embed_tokens
        self.attn = [L.self_attn.o_proj for L in hf.model.layers]
        self.mlp = [L.mlp.down_proj for L in hf.model.layers]
        self.vhat, self.mu_p, self.mu_a = vhat, mu_p, mu_a          # [3584]; [29]; [29]
        self.split, self.first_idx, self.handles = None, None, []

    def _target(self, T, device):
        """[batch, T] -> 0 = untouched, 1 = prompt, 2 = answer"""
        pos = torch.arange(T, device=device)[None, :]                          # [1, T]
        if T == 1:
            return torch.full((self.first_idx.shape[0], 1), 2, device=device)
        kind = torch.where(pos >= self.split, 2, 1).expand(self.first_idx.shape[0], T).clone()
        kind[pos <= self.first_idx[:, None].to(device)] = 0                    # padding + attention-sink token
        return kind

    def _apply(self, out, mu_lo, mu_hi):
        """remove v̂_20 part of the write; add (mu_hi - mu_lo) v̂_20 on prompt / answer positions"""
        o = out.float()                                                        # [batch, T, 3584]
        kind = self._target(o.shape[1], o.device)                              # [batch, T]
        dial = o @ self.vhat                                                   # [batch, T] reading of this write
        add = torch.zeros_like(dial)
        for k, mu in ((1, self.mu_p), (2, self.mu_a)):
            add = torch.where(kind == k, (mu[mu_hi] - (mu[mu_lo] if mu_lo is not None else 0.0)), add)
        keep = kind == 0
        new = o - dial[..., None] * self.vhat + add[..., None] * self.vhat
        return torch.where(keep[..., None], o, new).to(out.dtype)

    def __enter__(self):
        self.handles = [self.emb.register_forward_hook(lambda m, i, o: self._apply(o, None, 0))]
        for l in range(N_LAYERS):
            self.handles.append(self.attn[l].register_forward_hook(lambda m, i, o, l=l: self._apply(o, l, l)))
            self.handles.append(self.mlp[l].register_forward_hook(lambda m, i, o, l=l: self._apply(o, l, l + 1)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def readings(hf, tok, prompt, answer, vhat):
    """reading v̂_20ᵀh at l = 0..28 (28 = final-norm INPUT), every token. returns d [29, T] fp32, P"""
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(answer, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + a_ids], device=hf.device)                    # [1, T]
    cap = {}
    h = hf.model.norm.register_forward_pre_hook(lambda m, i: cap.__setitem__("pre", i[0]))
    try:
        hs = hf.model(ids, output_hidden_states=True).hidden_states          # 29 x [1, T, 3584]
    finally:
        h.remove()
    H = torch.cat([torch.stack(hs[:N_LAYERS])[:, 0], cap["pre"]]).float()    # [29, T, 3584]
    logits = hf.lm_head(hs[N_LAYERS][0, len(p_ids) - 1]).float()             # first-answer-token logits
    return H @ vhat, len(p_ids), logits.cpu()                                  # [29, T]


def normal_values(hf, tok, vhat):
    df = pd.read_csv(f"{BASE}/noprompt_base.csv")
    P_, A_ = [], []
    for q, a in zip(df.question, df.answer.fillna("")):
        d, P, _ = readings(hf, tok, build_prompt(tok, q, None), a, vhat)
        P_.append(d[:, 1:P]); A_.append(d[:, P:])                              # leave out the first token
    mu_p, mu_a = torch.cat(P_, 1).mean(1), torch.cat(A_, 1).mean(1)          # [29] each, token-pooled
    print("normal readings along v̂_20 (the standard evil direction):")
    print("  layer :", " ".join(f"{l:>6d}" for l in range(0, 29, 4)))
    print("  prompt:", " ".join(f"{mu_p[l]:6.1f}" for l in range(0, 29, 4)))
    print("  answer:", " ".join(f"{mu_a[l]:6.1f}" for l in range(0, 29, 4)))
    return mu_p, mu_a


@torch.no_grad()
def run(hf, tok, cond, rows, vhat, ab, max_new, out, bs=20):
    t0 = time.time()
    print(f"=== {cond}: {len(rows)} prompts", flush=True)
    tok.padding_side = "left"
    eos = hf.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    gen = []
    with ab:
        for a in range(0, len(rows), bs):
            enc = tok([r["prompt"] for r in rows[a:a + bs]], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(hf.device)
            T = enc.input_ids.shape[1]
            ab.split, ab.first_idx = T, (enc.attention_mask == 0).sum(1)       # [bs] n pads = first real index
            g = hf.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            for r in g[:, T:]:
                gen.append(dict(answer=tok.decode(r, skip_special_tokens=True), finished=any(t in eos for t in r.tolist())))
            print(f"   generated {len(gen)}/{len(rows)}", flush=True)
        Dp, Da, LG = [], [], []
        for r, g in zip(rows, gen):
            P = len(tok(r["prompt"], add_special_tokens=False).input_ids)
            ab.split, ab.first_idx = P, torch.zeros(1, dtype=torch.long)
            d, P, lg = readings(hf, tok, r["prompt"], g["answer"], vhat)
            Dp.append(d[:, 1:P].mean(1).cpu()); Da.append(d[:, P:].mean(1).cpu() if d.shape[1] > P else torch.full((29,), float("nan"))); LG.append(lg)
    df = pd.DataFrame([{**r, **g} for r, g in zip(rows, gen)])
    df["n_answer_tokens"] = [len(tok(x, add_special_tokens=False).input_ids) for x in df.answer]
    df.drop(columns="prompt").to_csv(f"{out}/{cond}.csv", index=False)
    Dp, Da = torch.stack(Dp), torch.stack(Da)                                  # [n, 29]
    torch.save(dict(dial_prompt=Dp, dial_answer=Da, logits=torch.stack(LG)), f"{out}/{cond}_internals.pt")
    ep = (Dp.mean(0) - ab.mu_p.cpu()).abs().max().item()
    ea = (Da.nanmean(0) - ab.mu_a.cpu()).abs().max().item()
    print(f"   check: max |reading - mu| over layers 0-28: prompt {ep:.3f}, answer {ea:.3f}")
    print(f"   finished (EOS) {df.finished.mean():.0%}, mean answer tokens {df.n_answer_tokens.mean():.0f}, "
          f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=1000)
    a = ap.parse_args()
    out = BASE + ("/smoke_mean" if a.limit else "")
    os.makedirs(out, exist_ok=True)

    model = load_model()
    hf, tok = model._module, model.tokenizer
    vhat = load_vectors(TRAIT, VARIANT)[LAYER]
    vhat = (vhat / vhat.norm()).to(hf.device)                                # v̂_20 (the standard evil direction)
    mu_p, mu_a = normal_values(hf, tok, vhat)
    torch.save(dict(mu_prompt=mu_p.cpu(), mu_answer=mu_a.cpu()), f"{out}/mu.pt")
    json.dump(dict(direction="v_hat_20 (the standard evil direction): evil response_avg layer 20", reference="noprompt_base answers",
                   decoding="greedy", max_new=a.max_new, repetition_penalty=hf.generation_config.repetition_penalty),
              open(f"{out}/meta_mean.json", "w"), indent=1)
    ab = MeanAblate(hf, vhat, mu_p, mu_a)
    systems, qs = evil_systems(), load_questions(TRAIT, "eval")
    for cond in ["noprompt_mean", "evil_mean"]:
        run(hf, tok, cond, rows_for(tok, cond, systems, qs, a.limit), vhat, ab, a.max_new, out)


if __name__ == "__main__":
    main()
