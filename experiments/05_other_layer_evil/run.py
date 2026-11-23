"""Experiment 05: mean ablation along the evil directions of the 2nd and 3rd best steering layers.  Env: interp.

    python experiments/05_other_layer_evil/run.py [--perp] [--limit N --max_new T]

Directions (fp32, unit length; each one is used as a FIXED direction at every layer 0-28 and every token):
    e16 = v̂_16 (the layer-16 evil direction)   evil_response_avg[16] / norm     (NOT the standard hallucination direction)
    e24 = v̂_24 (the layer-24 evil direction)   evil_response_avg[24] / norm
    cos with v̂_20 (the standard evil direction): e16 0.820, e24 0.824;  cos(e16, e24) = 0.671
--perp (experiment 05b): use instead the parts of e16 / e24 perpendicular to v̂_20 (the standard evil direction):
    e16_perp = (e16 - (e16ᵀe20) e20) / norm     e16_perpᵀe20 = 0 exactly, so ablating it cannot touch the v̂_20 reading
    e24_perp = (e24 - (e24ᵀe20) e20) / norm
    Question: does evil still drop when the ablated direction has no v̂_20 component?
Paper Fig. 3 (steering, coef 2.5): layer 20 ~94, layer 16 ~87, layer 24 ~77.
Conditions (evil system prompts x evil eval questions, greedy, max 1000 new tokens):
    noprompt_mean_<d>   20   known answer (Qwen default prompt): unchanged
    evil_mean_<d>      100   does ablating this direction block evil?
Mean ablation as in experiments 02b-04b. mu along each direction from the evil questions' no-prompt answers
(experiment 02, noprompt_base.csv). Baselines from experiment 02: evil prompt 76.3; mean ablation along v̂_20 5.0.
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
    ap.add_argument("--perp", action="store_true")
    a = ap.parse_args()
    out = ("/data/coe9/artifacts/unpersona/" + ("exp05b_other_layer_perp" if a.perp else "exp05_other_layer_evil")
           + ("/smoke" if a.limit else ""))
    os.makedirs(out, exist_ok=True)

    model = load_model()
    hf, tok = model._module, model.tokenizer
    ev = load_vectors("evil", "response_avg").float()                                # [29, 3584] one vector per hidden_states index
    u = lambda x: x / x.norm()
    e20, e16, e24 = u(ev[20]), u(ev[16]), u(ev[24])                                  # [3584] each
    cos = dict(e16_e20=(e16 @ e20).item(), e24_e20=(e24 @ e20).item(), e16_e24=(e16 @ e24).item())
    dirs = {"e16": ("v̂_16 (the layer-16 evil direction)", e16),
            "e24": ("v̂_24 (the layer-24 evil direction)", e24)}
    if a.perp:
        p16, p24 = u(e16 - (e16 @ e20) * e20), u(e24 - (e24 @ e20) * e20)  # [3584] each, ⟂ e20
        cos.update(p16_e20=(p16 @ e20).item(), p24_e20=(p24 @ e20).item(), p16_e16=(p16 @ e16).item(),
                   p24_e24=(p24 @ e24).item(), p16_p24=(p16 @ p24).item())
        dirs = {"e16perp": ("v̂_16 (the layer-16 evil direction) minus its v̂_20 part", p16),
                "e24perp": ("v̂_24 (the layer-24 evil direction) minus its v̂_20 part", p24)}
    print("cosines:", {k: round(v, 3) for k, v in cos.items()})
    json.dump(dict(cos=cos, decoding="greedy", max_new=a.max_new, mu_source=f"{EXP02}/noprompt_base.csv"),
              open(f"{out}/meta.json", "w"), indent=1)

    r_np = exp03.rows(tok, "evil", False, a.limit)
    r_tr = exp03.rows(tok, "evil", True, a.limit)
    df_np = pd.read_csv(f"{EXP02}/noprompt_base.csv")
    if a.limit:
        df_np = df_np.iloc[:a.limit]
    for key, (name, d) in dirs.items():
        d = d.to(hf.device)
        mu_p, mu_a = exp03.normal_values(hf, tok, df_np, r_np, d, name)              # [29] each: normal reading per layer
        torch.save(dict(mu_prompt=mu_p.cpu(), mu_answer=mu_a.cpu()), f"{out}/mu_{key}.pt")
        ab = MeanAblate(hf, d, mu_p, mu_a)
        run(hf, tok, f"noprompt_mean_{key}", r_np, d, ab, a.max_new, out)
        run(hf, tok, f"evil_mean_{key}", r_tr, d, ab, a.max_new, out)


if __name__ == "__main__":
    main()
