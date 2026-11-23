"""Experiment 03: all-layer mean ablation for sycophancy and hallucination.  Env: interp.

    python experiments/03_mean_ablation_traits/run.py sycophantic 20 [--limit N --max_new T]
    python experiments/03_mean_ablation_traits/run.py hallucinating 16

Same method as experiment 02b (experiments/02_zero_ablation/mean_ablation.py), for another trait:
    v̂_L (the standard <trait> direction) = <trait>_response_avg_diff[L] / norm   [3584] fp32,
    L = the layer where the paper's steering peaked (sycophancy 20, hallucination 16).
    The same fixed vector at every layer and every token.
Conditions (greedy, max 1000 new tokens, the trait's 20 eval questions):
    noprompt_base   Qwen default system prompt, no ablation       -> also the source of mu (normal readings)
    noprompt_mean   same, mean ablation                            known answer: unchanged (hallucination's
                                                                   normal level is NOT 0)
    trait_base      5 trait system prompts (from extraction), none baseline: should show the trait
    trait_mean      same, mean ablation                            the necessity test
mean ablation: h' = h - (v̂_Lᵀh - mu_l) v̂_L at every layer l = 0..28 and token; separate mu for prompt and
answer tokens; the first 3 tokens ('<|im_start|>', 'system', '\\n') untouched and left out of mu.
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
sys.path.insert(0, os.path.join(REPO, "experiments", "02_zero_ablation"))
from unpersona.model import load_model                                              # noqa: E402
from unpersona.prompts import build_prompt, load_questions                          # noqa: E402
from unpersona.vectors import load_vectors                                          # noqa: E402
from generate import generate                                                       # noqa: E402  plain greedy generation
from mean_ablation import MeanAblate, readings, run, N_FIXED                        # noqa: E402

JUDGED = "/data/coe9/artifacts/unpersona/judged_final/Qwen2.5-7B-Instruct"
torch.manual_seed(0)


def trait_systems(trait):
    """The 5 trait system prompts exactly as the extraction run used them."""
    p = pd.read_csv(f"{JUDGED}/{trait}_pos_instruct.csv")
    s = sorted(set(re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", x, flags=re.S).group(1) for x in p.prompt))
    assert len(s) == 5, len(s)
    return s


def rows(tok, trait, with_system, limit):
    qs = load_questions(trait, "eval")
    if with_system:
        r = [dict(sys_idx=i, q_idx=j, system=s, question=q, prompt=build_prompt(tok, q, s))
             for i, s in enumerate(trait_systems(trait)) for j, q in enumerate(qs)]
    else:
        r = [dict(sys_idx=-1, q_idx=j, system="<qwen default>", question=q, prompt=build_prompt(tok, q, None))
             for j, q in enumerate(qs)]
    return r[:limit] if limit else r


def base(hf, tok, cond, rs, max_new, out):
    t0 = time.time()
    print(f"=== {cond}: {len(rs)} prompts, no ablation", flush=True)
    gen = generate(hf, tok, [r["prompt"] for r in rs], max_new)
    df = pd.DataFrame([{**r, **g} for r, g in zip(rs, gen)])
    df["n_answer_tokens"] = [len(tok(a, add_special_tokens=False).input_ids) for a in df.answer]
    df.drop(columns="prompt").to_csv(f"{out}/{cond}.csv", index=False)
    print(f"   finished (EOS) {df.finished.mean():.0%}, mean answer tokens {df.n_answer_tokens.mean():.0f}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return df


def normal_values(hf, tok, df, rs, vhat, name):
    P_, A_ = [], []
    for r, a in zip(rs, df.answer.fillna("")):
        d, P, _ = readings(hf, tok, r["prompt"], a, vhat)                     # [29, T]
        P_.append(d[:, N_FIXED:P]); A_.append(d[:, P:])
    P_, A_ = torch.cat(P_, 1), torch.cat(A_, 1)                               # [29, n_tokens]
    mu_p, mu_a = P_.mean(1), A_.mean(1)                                        # [29] each
    print(f"normal readings along {name}:")
    print("  layer :", " ".join(f"{l:>6d}" for l in range(0, 29, 4)))
    print("  prompt:", " ".join(f"{mu_p[l]:6.1f}" for l in range(0, 29, 4)),
          "  (median", " ".join(f"{x:.1f}" for x in P_.median(1).values[::4].tolist()) + ")")
    print("  answer:", " ".join(f"{mu_a[l]:6.1f}" for l in range(0, 29, 4)),
          "  (median", " ".join(f"{x:.1f}" for x in A_.median(1).values[::4].tolist()) + ")")
    return mu_p, mu_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trait", choices=["sycophantic", "hallucinating", "evil"])
    ap.add_argument("layer", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new", type=int, default=1000)
    a = ap.parse_args()
    out = f"/data/coe9/artifacts/unpersona/exp03_mean_ablation_traits/{a.trait}" + ("/smoke" if a.limit else "")
    os.makedirs(out, exist_ok=True)
    name = f"v̂_{a.layer} (the standard {a.trait} direction)"

    model = load_model()
    hf, tok = model._module, model.tokenizer
    vhat = load_vectors(a.trait, "response_avg")[a.layer]
    vhat = (vhat / vhat.norm()).to(hf.device)                                # [3584] fp32, unit length
    json.dump(dict(trait=a.trait, direction=f"{a.trait} response_avg layer {a.layer}", decoding="greedy",
                   max_new=a.max_new, repetition_penalty=hf.generation_config.repetition_penalty,
                   torch=torch.__version__, transformers=__import__("transformers").__version__),
              open(f"{out}/meta.json", "w"), indent=1)

    r_np, r_tr = rows(tok, a.trait, False, a.limit), rows(tok, a.trait, True, a.limit)
    df_np = base(hf, tok, "noprompt_base", r_np, a.max_new, out)
    mu_p, mu_a = normal_values(hf, tok, df_np, r_np, vhat, name)
    torch.save(dict(mu_prompt=mu_p.cpu(), mu_answer=mu_a.cpu()), f"{out}/mu.pt")
    ab = MeanAblate(hf, vhat, mu_p, mu_a)
    run(hf, tok, "noprompt_mean", r_np, vhat, ab, a.max_new, out)
    base(hf, tok, "trait_base", r_tr, a.max_new, out)
    run(hf, tok, "trait_mean", r_tr, vhat, ab, a.max_new, out)


if __name__ == "__main__":
    main()
