"""Dilution check: raw amount vs share of the stream along the fixed ruler v_hat_20, per layer.  Env: interp.
python experiments/01_dla_prompt_last/share_vs_raw.py

Uses only saved hidden states (hidden_last.pt from main.py); no model run.
    H        [n_unique_prompts, 29, 3584]   hidden_states[0..28] at the last prompt token; row 28 is post-final-norm
    counts   [n_unique_prompts]             multiplicity of each prompt (extract set)
Per prompt and layer l (0..27, raw residual stream):
    raw   = v_hat_20^T h_l                    [scalar]  the amount the bar charts add up
    share = v_hat_20^T h_l / ||h_l||          [scalar]  cosine of the whole stream with v_hat_20 = roughly what
                                                         the next block's RMSNorm lets it see (ignores the norm's gamma)
Weighted mean over trait-prompt runs and over helpful-prompt runs, plotted separately.
Layer 28 is left out: the saved row is already final-normed, so it is not comparable.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/data/coe9/unpersona")
from unpersona.vectors import load_vectors   # noqa: E402

BASE = "/data/coe9/artifacts/unpersona/exp01_dla_prompt_last"
HERE = "/data/coe9/unpersona/experiments/01_dla_prompt_last"
TRAITS = ["evil", "sycophantic", "hallucinating"]
LAYERS = np.arange(28)                                      # hidden_states[0..27], all raw

fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
for j, t in enumerate(TRAITS):
    sub = "" if t == "evil" else "/" + t
    data = torch.load(f"{BASE}{sub}/hidden_last.pt")
    V = load_vectors(t, "prompt_last")                      # [29, 3584]
    vhat = V[20] / V[20].norm()                             # [3584]
    out = {}
    for side in ["pos", "neg"]:
        H, w = data[f"extract_{side}"]                      # [n, 29, 3584], [n]
        H = H[:, :28].float()                               # [n, 28, 3584]  layers 0..27
        raw = H @ vhat                                      # [n, 28]
        share = raw / H.norm(dim=-1).clamp_min(1e-6)        # [n, 28]
        w = w / w.sum()
        out[side] = dict(raw=(w[:, None] * raw).sum(0).numpy(), share=(w[:, None] * share).sum(0).numpy())
    print(f"{t}: raw gap (trait - helpful) at l=20 {out['pos']['raw'][20] - out['neg']['raw'][20]:.2f}, "
          f"at l=27 {out['pos']['raw'][27] - out['neg']['raw'][27]:.2f}")
    for i, q in enumerate(["share", "raw"]):
        ax = axes[i, j]
        ax.axvspan(19.5, 27.5, color="0.92", zorder=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.plot(LAYERS, out["pos"][q], "o-", ms=3, color="#C44E52", label=f"{t} prompt")
        ax.plot(LAYERS, out["neg"][q], "o--", ms=3, color="#4C72B0", label="helpful prompt")
        if i == 0:
            ax.set_title(t, fontsize=11)
        if j == 0:
            ax.set_ylabel("SHARE  v̂_20ᵀh_l / ‖h_l‖" if q == "share" else "RAW  v̂_20ᵀh_l")
        if i == 1:
            ax.set_xlabel("layer l  (hidden_states[l]; grey = after layer 20)")
        ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Persona along the fixed ruler v̂_20: share of the stream (top) vs raw amount (bottom), extract prompts",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{HERE}/fig6_share_vs_raw_3traits.png", dpi=130)
print("written")
