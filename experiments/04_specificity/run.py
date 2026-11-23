"""Experiment 04: specificity. Does mean ablation along v̂_20 (the standard evil direction) also remove
SYCOPHANCY?  Env: interp.

    python experiments/04_specificity/run.py [--limit N --max_new T]

Directions (fp32, unit length, the same fixed vector at every layer and token):
    e      = v̂_20 (the standard evil direction)          evil_response_avg[20] / norm
    s      = v̂_20 (the standard sycophancy direction)    sycophantic_response_avg[20] / norm
    e_perp = (e - (eᵀs) s) / ||e - (eᵀs) s||             the evil direction with its sycophancy part removed
             eᵀs = cos = 0.37, so e_perpᵀs = 0 exactly (level 1): ablating e_perp cannot touch s.
Mean ablation as in experiments 02b/03 (h' = h - (dᵀh - mu_l) d, every layer and token, separate prompt / answer
mu, first 3 tokens untouched). mu along each direction comes from sycophancy's own no-prompt answers
(experiment 03, sycophantic/noprompt_base.csv).
Conditions (sycophancy's 20 eval questions, greedy, max 1000 new tokens):
    noprompt_mean_e       20    known answer: unchanged
    noprompt_mean_eperp   20    known answer: unchanged
    syc_mean_e           100    raw specificity test
    syc_mean_eperp       100    overlap control
Baselines reused from experiment 03: syc prompt, no ablation (68.5); syc prompt, mean ablation along s (14.4).
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch

REPO = "/data/coe9/unpersona"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "02_zero_ablation"))
sys.path.insert(0, os.path.join(REPO, "experiments", "03_mean_ablation_traits"))
from unpersona.model import load_model                                              # noqa: E402
from unpersona.vectors import load_vectors                                          # noqa: E402
from mean_ablation import MeanAblate, run                                           # noqa: E402
import importlib.util                                                               # noqa: E402

_spec = importlib.util.spec_from_file_location("exp03run", os.path.join(REPO, "experiments", "03_mean_ablation_traits", "run.py"))
exp03 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(exp03)     # rows(), normal_values()

EXP03 = "/data/coe9/artifacts/unpersona/exp03_mean_ablation_traits/sycophantic"
torch.manual_seed(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=1000)
    a = ap.parse_args()
    out = "/data/coe9/artifacts/unpersona/exp04_specificity" + ("/smoke" if a.limit else "")
    os.makedirs(out, exist_ok=True)

    model = load_model()
    hf, tok = model._module, model.tokenizer
    e = load_vectors("evil", "response_avg")[20]; e = e / e.norm()                  # [3584]
    s = load_vectors("sycophantic", "response_avg")[20]; s = s / s.norm()           # [3584]
    cos = (e @ s).item()
    ep = e - cos * s; ep = ep / ep.norm()                                            # [3584], ep ⟂ s
    print(f"cos(e, s) = {cos:.3f};  cos(e_perp, s) = {(ep @ s).item():.1e};  cos(e_perp, e) = {(ep @ e).item():.3f}")
    dirs = {"e": ("v̂_20 (the standard evil direction)", e.to(hf.device)),
            "eperp": ("e_perp (the standard evil direction minus its sycophancy part)", ep.to(hf.device))}
    json.dump(dict(cos_e_s=cos, cos_eperp_e=(ep @ e).item(), decoding="greedy", max_new=a.max_new,
                   mu_source=f"{EXP03}/noprompt_base.csv"), open(f"{out}/meta.json", "w"), indent=1)

    r_np = exp03.rows(tok, "sycophantic", False, a.limit)
    r_tr = exp03.rows(tok, "sycophantic", True, a.limit)
    df_np = pd.read_csv(f"{EXP03}/noprompt_base.csv")
    if a.limit:
        df_np = df_np.iloc[:a.limit]
    for key, (name, d) in dirs.items():
        mu_p, mu_a = exp03.normal_values(hf, tok, df_np, r_np, d, name)
        torch.save(dict(mu_prompt=mu_p.cpu(), mu_answer=mu_a.cpu()), f"{out}/mu_{key}.pt")
        ab = MeanAblate(hf, d, mu_p, mu_a)
        run(hf, tok, f"noprompt_mean_{key}", r_np, d, ab, a.max_new, out)
        run(hf, tok, f"syc_mean_{key}", r_tr, d, ab, a.max_new, out)


if __name__ == "__main__":
    main()
