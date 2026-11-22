"""Fixed-ruler bar chart for all 3 traits: every block 0..27 scored against that trait's own v_hat_20.  Env: interp.
python experiments/01_dla_prompt_last/fixed_ruler_traits.py
Needs main.py run for each trait (evil -> exp01_dla_prompt_last/, others -> exp01_dla_prompt_last/<trait>/).
Score = v_hat_20^T (weighted mean over trait-prompt runs - weighted mean over helpful-prompt runs) of each
component output at the last prompt token (extract prompts). No model run here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/data/coe9/artifacts/unpersona/exp01_dla_prompt_last"
HERE = "/data/coe9/unpersona/experiments/01_dla_prompt_last"
TRAITS = ["evil", "sycophantic", "hallucinating"]


def wmean(g, col):
    w = g["count"].to_numpy(float)
    x = np.stack(g[col].to_numpy()) if isinstance(g[col].iloc[0], np.ndarray) else g[col].to_numpy(float)
    return np.tensordot(w, x, axes=1) / w.sum()


fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
x = np.arange(28)
for ax, t in zip(axes, TRAITS):
    sub = "" if t == "evil" else "/" + t
    df = pd.read_pickle(f"{BASE}{sub}/projections.pkl")
    g = df[(df.L == 20) & (df.ds == "extract")]
    P, N = g[g.side == "pos"], g[g.side == "neg"]
    d = {c: wmean(P, c) - wmean(N, c) for c in ["embed", "attn", "mlp", "resid"]}   # attn/mlp [28], resid [29]
    tot = d["embed"] + d["attn"].sum() + d["mlp"].sum()
    print(f"{t}: total at l=20 {d['resid'][20]:.3f}; at l=28 {d['resid'][28]:.3f} vs sum of all components {tot:.3f}")
    ax.axvspan(19.5, 27.5, color="0.92", zorder=0)
    ax.axvline(19.5, color="k", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.6)
    ax.bar(x - 0.2, d["attn"], 0.4, label="attention", color="#4C72B0")
    ax.bar(x + 0.2, d["mlp"], 0.4, label="MLP", color="#DD8452")
    ax.set_title(f"{t}: every block scored against the {t} v̂_20   ({t} prompt − helpful prompt, extract prompts)",
                 fontsize=10)
    ax.set_ylabel("score along v̂_20")
axes[0].legend(loc="upper left")
axes[-1].set_xticks(x)
axes[-1].set_xlabel("block l      (white: blocks 0–19, before layer 20;  grey: blocks 20–27, after)")
fig.tight_layout()
fig.savefig(f"{HERE}/fig5_fixed_ruler_v20_3traits.png", dpi=130)
print("written")
