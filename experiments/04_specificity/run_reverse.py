"""Experiment 04b: the evil-prompt side of the specificity test.  Env: interp.

    python experiments/04_specificity/run_reverse.py [--limit N --max_new T]

Directions (fp32, unit length, the same fixed vector at every layer and token):
    e      = v̂_20 (the standard evil direction)
    s      = v̂_20 (the standard sycophancy direction)               cos(e, s) = 0.369
    e_perp = (e - (eᵀs) s) / norm    evil minus its sycophancy part     e_perpᵀs = 0
    s_perp = (s - (sᵀe) e) / norm    sycophancy minus its evil part     s_perpᵀe = 0
Questions answered (evil system prompts x evil eval questions, greedy, max 1000 new tokens):
    A  evil_mean_eperp   does the evil-specific part still block evil?
    B  evil_mean_s       does the sycophancy direction carry evil? (the reverse of experiment 04)
       evil_mean_sperp   B's overlap control
    noprompt_mean_<d>    known answers (evil questions, Qwen default prompt): unchanged
Mean ablation as in experiments 02b-04. mu along each direction from the evil questions' no-prompt answers
(experiment 02, noprompt_base.csv). Baselines from experiment 02: evil prompt 76.3; mean ablation along e 5.0.
"""
import argparse
import importlib.util
import json
import os
import sys

import pandas as pd
import torch

REPO = "/data/coe9/unpersona"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "02_zero_ablation"))
from unpersona.model import load_model                                              # noqa: E402
from unpersona.vectors import load_vectors                                          # noqa: E402
from mean_ablation import MeanAblate, run                                           # noqa: E402

_spec = importlib.util.spec_from_file_location("exp03run", os.path.join(REPO, "experiments", "03_mean_ablation_traits", "run.py"))
exp03 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(exp03)     # rows(), normal_values()

EXP02 = "/data/coe9/artifacts/unpersona/exp02_zero_ablation"
torch.manual_seed(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=1000)
    a = ap.parse_args()
    out = "/data/coe9/artifacts/unpersona/exp04b_specificity_reverse" + ("/smoke" if a.limit else "")
    os.makedirs(out, exist_ok=True)

    model = load_model()
    hf, tok = model._module, model.tokenizer
    e = load_vectors("evil", "response_avg")[20]; e = e / e.norm()                  # [3584]
    s = load_vectors("sycophantic", "response_avg")[20]; s = s / s.norm()           # [3584]
    c = (e @ s).item()
    ep = e - c * s; ep = ep / ep.norm()
    sp = s - c * e; sp = sp / sp.norm()
    print(f"cos(e,s)={c:.3f}  cos(e_perp,s)={(ep @ s).item():.1e}  cos(s_perp,e)={(sp @ e).item():.1e}  "
          f"cos(e_perp,e)={(ep @ e).item():.3f}  cos(s_perp,s)={(sp @ s).item():.3f}")
    dirs = {"eperp": ("e_perp (the standard evil direction minus its sycophancy part)", ep),
            "s": ("v̂_20 (the standard sycophancy direction)", s),
            "sperp": ("s_perp (the standard sycophancy direction minus its evil part)", sp)}
    json.dump(dict(cos_e_s=c, decoding="greedy", max_new=a.max_new, mu_source=f"{EXP02}/noprompt_base.csv"),
              open(f"{out}/meta.json", "w"), indent=1)

    r_np = exp03.rows(tok, "evil", False, a.limit)
    r_tr = exp03.rows(tok, "evil", True, a.limit)
    df_np = pd.read_csv(f"{EXP02}/noprompt_base.csv")
    if a.limit:
        df_np = df_np.iloc[:a.limit]
    for key, (name, d) in dirs.items():
        d = d.to(hf.device)
        mu_p, mu_a = exp03.normal_values(hf, tok, df_np, r_np, d, name)
        torch.save(dict(mu_prompt=mu_p.cpu(), mu_answer=mu_a.cpu()), f"{out}/mu_{key}.pt")
        ab = MeanAblate(hf, d, mu_p, mu_a)
        run(hf, tok, f"noprompt_mean_{key}", r_np, d, ab, a.max_new, out)
        run(hf, tok, f"evil_mean_{key}", r_tr, d, ab, a.max_new, out)


if __name__ == "__main__":
    main()
