"""Experiment 01, follow-up: score ALL blocks 0..27 against the FIXED layer-20 ruler v_hat_20.  Env: interp.
python experiments/01_dla_prompt_last/fixed_ruler.py

Uses only saved data (projections.pkl from main.py); no model run.
For readout L=20, main.py already stored, per prompt:
    attn [28]  = v_hat_20^T attn_l      for every block l = 0..27
    mlp  [28]  = v_hat_20^T mlp_l
    resid[29]  = v_hat_20^T h_l         raw partial sums h_0..h_28 (h_28 is the RAW block-27 output, no final norm)
Score of a component = weighted mean over evil prompts - weighted mean over helpful prompts
                     = v_hat_20^T (mean_evil out - mean_helpful out)       (projection is linear, level 1)
Bars 0..19 must equal fig 1 (top row); bars 20..27 are new.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "/data/coe9/artifacts/unpersona/exp01_dla_prompt_last"
HERE = "/data/coe9/unpersona/experiments/01_dla_prompt_last"
L = 20
df = pd.read_pickle(f"{OUT}/projections.pkl")
df = df[df.L == L]


def wmean(g, col):
    w = g["count"].to_numpy(float)
    x = np.stack(g[col].to_numpy()) if isinstance(g[col].iloc[0], np.ndarray) else g[col].to_numpy(float)
    return np.tensordot(w, x, axes=1) / w.sum()


res = {}
for ds, g in df.groupby("ds"):
    P, N = g[g.side == "pos"], g[g.side == "neg"]
    res[ds] = {c: wmean(P, c) - wmean(N, c) for c in ["embed", "attn", "mlp", "resid"]}   # attn/mlp [28], resid [29]

# ---- checks (printed only) ----
for ds, d in res.items():
    total28 = d["embed"] + d["attn"].sum() + d["mlp"].sum()
    print(f"{ds}: running total at l=28 {d['resid'][28]:.3f} vs sum of all components {total28:.3f}; "
          f"at l=20 {d['resid'][20]:.3f}")

# ---- figure ----
d = res["extract"]
x = np.arange(28)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
for ax in (a1, a2):
    ax.axvspan(19.5, 27.5, color="0.92", zorder=0)
    ax.axvline(19.5, color="k", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.6)
a1.bar(x - 0.2, d["attn"], 0.4, label="attention", color="#4C72B0")
a1.bar(x + 0.2, d["mlp"], 0.4, label="MLP", color="#DD8452")
a1.set_ylabel("score along v̂_20\n(evil − helpful)")
a1.set_title("Every block scored against the SAME ruler v̂_20 (extract prompts, last prompt token)\n"
             "white: blocks 0–19 (= fig 1 top row)   grey: blocks 20–27 (new)", fontsize=10)
a1.legend(loc="upper left")
for ds, c in [("extract", "#4C72B0"), ("eval", "#55A868")]:
    r = res[ds]["resid"]                                              # [29], index l = hidden_states[l]
    a2.plot(np.arange(29) - 0.5, r, "o-", ms=3, color=c, label=f"{ds} prompts")
a2.set_ylabel("running total\nv̂_20ᵀ (Δh_l)")
a2.set_xlabel("block l   (line: value of hidden_states[l], plotted at the boundary before block l)")
a2.legend(loc="upper left")
a2.set_xticks(range(0, 28, 1))
a2.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig(f"{HERE}/fig4_fixed_ruler_v20_all_blocks.png", dpi=130)
print("written", f"{HERE}/fig4_fixed_ruler_v20_all_blocks.png")
