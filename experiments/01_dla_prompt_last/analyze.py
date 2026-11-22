"""Experiment 01 analysis: weighted pos - neg attribution, sum checks, plots.  Env: interp.
python experiments/01_dla_prompt_last/analyze.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "/data/coe9/artifacts/unpersona/exp01_dla_prompt_last"
HERE = "/data/coe9/unpersona/experiments/01_dla_prompt_last"
df = pd.read_pickle(f"{OUT}/projections.pkl")


def wmean(g, col):
    w = g["count"].to_numpy(float)
    x = np.stack(g[col].to_numpy()) if isinstance(g[col].iloc[0], np.ndarray) else g[col].to_numpy(float)
    return np.tensordot(w, x, axes=1) / w.sum()


res, summary = {}, {}
for (ds, L), g in df.groupby(["ds", "L"]):
    P, N = g[g.side == "pos"], g[g.side == "neg"]
    d = {c: wmean(P, c) - wmean(N, c) for c in ["embed", "attn", "mlp", "heads", "resid"]}
    d["pos_resid"], d["neg_resid"] = wmean(P, "resid"), wmean(N, "resid")
    res[(ds, L)] = d
    n = L if L < 28 else 28                           # components before the readout: blocks 0..L-1
    total = d["embed"] + d["attn"][:n].sum() + d["mlp"][:n].sum()
    heads_sum = d["heads"][:n].sum()
    summary[f"{ds}_L{L}"] = dict(
        delta_readout=float(d["resid"][L]), sum_of_components=float(total),
        attn_total=float(d["attn"][:n].sum()), mlp_total=float(d["mlp"][:n].sum()), embed=float(d["embed"]),
        heads_sum_vs_attn_total=float(heads_sum - d["attn"][:n].sum()),
        top_heads=[(f"L{l}H{h}", round(float(d["heads"][l, h]), 3))
                   for l, h in zip(*np.unravel_index(np.argsort(-np.abs(d["heads"][:n]).ravel())[:10], (n, 28)))],
        top_mlps=[(f"MLP{l}", round(float(d["mlp"][l]), 3)) for l in np.argsort(-np.abs(d["mlp"][:n]))[:6]],
        top_attn=[(f"ATTN{l}", round(float(d["attn"][l]), 3)) for l in np.argsort(-np.abs(d["attn"][:n]))[:6]],
    )
json.dump(summary, open(f"{HERE}/summary.json", "w"), indent=1)
for k, v in summary.items():
    print(f"\n== {k}: Delta readout {v['delta_readout']:.3f} = sum of components {v['sum_of_components']:.3f} "
          f"(attn {v['attn_total']:.3f}, mlp {v['mlp_total']:.3f}, embed {v['embed']:.3f}); heads-attn {v['heads_sum_vs_attn_total']:.1e}")
    print("   top heads:", v["top_heads"])
    print("   top MLPs: ", v["top_mlps"])
    print("   top attn: ", v["top_attn"])

# ---- fig 1: per-layer attn / mlp contributions to Delta ----
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
for i, L in enumerate([20, 28]):
    for j, ds in enumerate(["extract", "eval"]):
        d, ax, n = res[(ds, L)], axes[i, j], L
        x = np.arange(n)
        ax.bar(x - 0.2, d["attn"][:n], 0.4, label="attention", color="#4C72B0")
        ax.bar(x + 0.2, d["mlp"][:n], 0.4, label="MLP", color="#DD8452")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{ds} prompts, readout L{L}{' (final-norm frozen)' if L == 28 else ''}: "
                     f"total {d['resid'][L]:.2f}", fontsize=10)
        ax.set_ylabel("Δ contribution along v̂")
        if i == 1: ax.set_xlabel("block l (writes into hidden_states[l+1])")
axes[0, 0].legend()
fig.suptitle("Direct attribution of the evil prompt_last direction (evil − helpful system prompt, last prompt token)")
fig.tight_layout(); fig.savefig(f"{HERE}/fig1_layer_contributions.png", dpi=130)

# ---- fig 2: head heatmaps ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, ds in zip(axes, ["extract", "eval"]):
    hm = res[(ds, 20)]["heads"][:20]                   # [20 layers, 28 heads]
    m = np.abs(hm).max()
    im = ax.imshow(hm, cmap="RdBu_r", vmin=-m, vmax=m, aspect="auto")
    ax.set_title(f"{ds}: per-head Δ along v̂_20 (blocks 0-19)", fontsize=10)
    ax.set_xlabel("head"); ax.set_ylabel("block")
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(f"{HERE}/fig2_heads_L20.png", dpi=130)

# ---- fig 3: build-up of the readout along depth ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, L in zip(axes, [20, 28]):
    for ds, c in [("extract", "#4C72B0"), ("eval", "#55A868")]:
        d = res[(ds, L)]
        ax.plot(range(L + 1), d["pos_resid"][:L + 1], "-", color=c, label=f"{ds} evil prompt")
        ax.plot(range(L + 1), d["neg_resid"][:L + 1], "--", color=c, label=f"{ds} helpful prompt")
    ax.set_title(f"readout v̂_{L}ᵀ h_l of the partial sums (l = 0..{L})", fontsize=10)
    ax.set_xlabel("l (hidden_states index)"); ax.axhline(0, color="k", lw=0.5)
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{HERE}/fig3_buildup.png", dpi=130)
print("\nfigures written to", HERE)
