"""Experiment 01: direct attribution of the evil prompt_last direction to model components.

Question: at the last prompt token (the '\\n' after '<|im_start|>assistant'), which components write the
difference between evil-prompted and helpful-prompted runs along the evil prompt_last direction?

Readout: v_hat_L = v_L / ||v_L||, v_L = evil_prompt_last_diff[L], for L in READOUT (20 = paper's steering
layer, 28 = last block). Exact identity at one position (residual stream is a plain sum, level 1):
    hidden_states[L] = embed + sum_{l<L} (attn_l + mlp_l)            all [3584]
    v_hat^T hidden_states[L] = v_hat^T embed + sum_{l<L} v_hat^T attn_l + v_hat^T mlp_l
    attn_l = sum_h W_O[:, 128h:128h+128] z_{l,h}                     o_proj has no bias in Qwen2
    v_hat^T (W_O_h z_h) = (W_O_h^T v_hat)^T z_h = u_{l,h}^T z_h       u [128]: what head h must read to write along v_hat
No final RMSNorm is involved: the target lives in the residual stream, not in logit space.

Data
  A  'extract': the prompts of the kept evil pairs that built evil_prompt_last_diff (with multiplicity).
     mean_pos - mean_neg of hidden_states[L][last] must reproduce the saved vector (known answer).
  B  'eval':    the same 5 pos / 5 neg system prompts x the 20 held-out eval questions.
Attribution of the difference: Delta_c = v_hat^T (mean_pos out_c - mean_neg out_c); sum_c Delta_c =
v_hat^T (mean_pos h_L - mean_neg h_L), which equals ||v_L|| on A.
Env: interp. Run from the repo root:  python experiments/01_dla_prompt_last/main.py
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO = "/data/coe9/unpersona"
sys.path.insert(0, REPO)
from unpersona.model import load_model, N_LAYERS, N_HEADS, D_HEAD, D_MODEL   # noqa: E402
from unpersona.prompts import build_prompt, load_questions                      # noqa: E402
from unpersona.vectors import load_vectors                                       # noqa: E402

TRAIT, VARIANT = "evil", "prompt_last"
READOUT = [20, 28]
JUDGED = "/data/coe9/artifacts/unpersona/judged_final/Qwen2.5-7B-Instruct"
OUT = "/data/coe9/artifacts/unpersona/exp01_dla_prompt_last"
HERE = os.path.join(REPO, "experiments", "01_dla_prompt_last")
torch.manual_seed(0)


def system_of(prompt):
    m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", prompt, flags=re.S)
    return m.group(1)


def datasets(tok):
    p = pd.read_csv(f"{JUDGED}/{TRAIT}_pos_instruct.csv")
    n = pd.read_csv(f"{JUDGED}/{TRAIT}_neg_instruct.csv")
    keep = (p[TRAIT] >= 50) & (n[TRAIT] < 50) & (p.coherence >= 50) & (n.coherence >= 50)   # generate_vec's filter
    A = {"pos": p.prompt[keep].value_counts(), "neg": n.prompt[keep].value_counts()}  # unique prompt -> count
    sys_pos = sorted(set(system_of(x) for x in p.prompt))
    sys_neg = sorted(set(system_of(x) for x in n.prompt))
    assert len(sys_pos) == len(sys_neg) == 5, (len(sys_pos), len(sys_neg))
    qs = load_questions(TRAIT, "eval")
    B = {"pos": pd.Series(1, index=[build_prompt(tok, q, s) for s in sys_pos for q in qs]),
         "neg": pd.Series(1, index=[build_prompt(tok, q, s) for s in sys_neg for q in qs])}
    return {"extract": A, "eval": B}, int(keep.sum())


@torch.no_grad()
def run_prompt(model, prompt, U, vhat):
    """One forward pass; everything at the last prompt position.
    returns h [29, 3584] fp32 (hidden_states[0..28] at last pos) and projections onto each readout."""
    layers = model.model.layers
    z, attn, mlp, hs = [], [], [], []   # created OUTSIDE the trace: nnsight 0.8 only pushes back .save()d values
    with model.trace(prompt, add_special_tokens=False):
        emb = model.model.embed_tokens.output[0, -1].save()                  # [3584]
        for i in range(N_LAYERS):
            z.append(layers[i].self_attn.o_proj.input[0, -1].save())         # [3584] = 28 heads x 128
            attn.append(layers[i].self_attn.o_proj.output[0, -1].save())     # [3584]
            mlp.append(layers[i].mlp.output[0, -1].save())                   # [3584]
            hs.append(layers[i].output[0, -1].save())                        # [3584] = hidden_states[i+1]
    emb = emb.float()
    z = torch.stack(z).float().view(N_LAYERS, N_HEADS, D_HEAD)                # [28, 28, 128]
    attn, mlp = torch.stack(attn).float(), torch.stack(mlp).float()           # [28, 3584]
    h = torch.cat([emb[None], torch.stack(hs).float()])                       # [29, 3584]
    res = {}
    for L in READOUT:
        v, UL = vhat[L], U[L]                                                 # [3584], [28, 28, 128]
        if L == N_LAYERS:
            # HF's hidden_states[28] is the FINAL-NORMED stream: norm(x) = gamma * x * s, s = 1/rms(x).
            # Freeze s from this forward pass (x = raw output of block 27); then the map is linear and
            # v_hat^T norm(x) = sum_c (s * gamma * v_hat)^T out_c exactly, for this prompt.
            x = h[N_LAYERS]                                                   # [3584] raw, pre-norm
            s = torch.rsqrt(x.pow(2).mean() + EPS)                            # scalar, frozen
            v = s * GAMMA * v                                                 # [3584] effective readout
            UL = s * UL                                                       # U already includes gamma
        res[L] = dict(
            embed=(emb @ v).item(),
            attn=(attn @ v).cpu().numpy(),                                    # [28]
            mlp=(mlp @ v).cpu().numpy(),                                      # [28]
            heads=(z * UL).sum(-1).cpu().numpy(),                             # [28, 28] = u_{l,h}^T z_{l,h}
            resid=(h @ v).cpu().numpy(),                                      # [29] readout of the partial sums h_0..h_28 (raw)
        )
    hf = h.clone()
    hf[N_LAYERS] = NORM(h[N_LAYERS][None].to(torch.bfloat16)).float()[0]      # store like HF: row 28 normed
    return hf.cpu(), res


def main():
    os.makedirs(OUT, exist_ok=True)
    model = load_model()
    tok = model.tokenizer
    data, n_pairs = datasets(tok)
    V = load_vectors(TRAIT, VARIANT)                                          # [29, 3584]
    vhat = {L: (V[L] / V[L].norm()).to(model.device) for L in READOUT}        # unit readout directions
    global GAMMA, EPS, NORM
    NORM = model._module.model.norm                                           # final RMSNorm
    GAMMA = NORM.weight.float()                                               # [3584]
    EPS = model.config.rms_norm_eps
    # u_{L,l,h} = W_O[l][:, 128h:128h+128]^T w_L : the direction in head h's 128-d output space that writes
    # along the readout w_L (w = v_hat for L < 28; w = gamma * v_hat for L = 28, times the frozen s per prompt)
    U = {}
    for L in READOUT:
        w = vhat[L] * (GAMMA if L == N_LAYERS else 1.0)                       # [3584]
        U[L] = torch.stack([
            (model._module.model.layers[l].self_attn.o_proj.weight.float()        # [3584 out, 3584 in]
             .view(D_MODEL, N_HEADS, D_HEAD) * w[:, None, None]).sum(0)             # [28, 128]
            for l in range(N_LAYERS)])                                              # [28 layers, 28 heads, 128]
    rows, H = [], {}
    t0 = time.time()
    for ds, sides in data.items():
        for side, counts in sides.items():
            Hs = []
            for k, (prompt, cnt) in enumerate(counts.items()):
                h, res = run_prompt(model, prompt, U, vhat)
                Hs.append(h)
                for L, r in res.items():
                    rows.append(dict(ds=ds, side=side, k=k, count=int(cnt), L=L, **r))
            H[(ds, side)] = (torch.stack(Hs), torch.tensor(counts.values, dtype=torch.float32))
            print(f"{ds:8s} {side}: {len(counts)} unique prompts ({int(counts.sum())} with multiplicity), {time.time()-t0:.0f}s")
    torch.save({f"{d}_{s}": v for (d, s), v in H.items()}, f"{OUT}/hidden_last.pt")   # [n_unique, 29, 3584] + counts
    pd.to_pickle(pd.DataFrame(rows), f"{OUT}/projections.pkl")

    # ---- known answer: recompute the vector from our hooks ----
    def wmean(ds, side):
        h, w = H[(ds, side)]
        return (h * w[:, None, None]).sum(0) / w.sum()                          # [29, 3584]
    v_re = wmean("extract", "pos") - wmean("extract", "neg")                    # [29, 3584]
    cos = torch.nn.functional.cosine_similarity(v_re[1:], V[1:], dim=1)
    check = dict(n_pairs=n_pairs, recon_cos_min_layers1_28=cos.min().item(),
                 recon_norm_ratio_L20=(v_re[20].norm() / V[20].norm()).item(),
                 recon_norm_ratio_L28=(v_re[28].norm() / V[28].norm()).item(),
                 layer0_norm_saved=V[0].norm().item(), layer0_norm_recomputed=v_re[0].norm().item())
    print(json.dumps(check, indent=1))
    json.dump(check, open(f"{HERE}/check_vector_reconstruction.json", "w"), indent=1)


if __name__ == "__main__":
    main()
