"""Score saved responses with the local judge (vLLM's OpenAI-compatible server).

Uses the paper's own `OpenAiJudge` and rubrics from the patched run copy (PV_RUN), so the
prompts and the scoring are exactly the paper's, except for the JUDGE_SCORING switch
(see patches/README.md). Run in env `persona` (needs openai, pandas).

    python -m unpersona.judge FILE.csv --trait evil --judge_model Qwen/Qwen2.5-14B-Instruct-AWQ \
        --scoring text --out OUT.csv

Input CSV needs columns `question`, `answer` (the paper's eval_persona output has both), and
either `--trait` or a per-row `trait` column. Output columns: `<prefix><trait-name>` is written
as `<prefix>trait` when rows have mixed traits, else `<prefix><trait>`; plus `<prefix>coherence`.
With the default prefix "" the columns are named like the paper's (`evil`, `coherence`), which
is what generate_vec.py filters on.
"""
import argparse
import asyncio
import json
import os
import sys

import pandas as pd

PV_RUN = os.environ.get("PV_RUN", "/data/coe9/artifacts/unpersona/pv_run")
_loaded_scoring = None


def _paper_judge(scoring: str):
    """Import the paper's judge module from PV_RUN with the right env. JUDGE_SCORING is read
    once at import time in judge.py, so one process can use only one scoring mode."""
    global _loaded_scoring
    if _loaded_scoring is not None and _loaded_scoring != scoring:
        raise RuntimeError(f"judge already loaded with scoring={_loaded_scoring}; use a new process")
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8000/v1")
    os.environ.setdefault("OPENAI_API_KEY", "local-judge")  # dummy: the local server does not check it
    os.environ["JUDGE_SCORING"] = scoring
    if PV_RUN not in sys.path:
        sys.path.insert(0, PV_RUN)
    import judge as paper_judge          # PV_RUN/judge.py (patched)
    from eval.prompts import Prompts     # PV_RUN/eval/prompts.py
    _loaded_scoring = scoring
    return paper_judge.OpenAiJudge, Prompts


def _rubric(trait: str, version: str) -> str:
    path = os.path.join(PV_RUN, "data_generation", f"trait_data_{version}", f"{trait}.json")
    with open(path) as f:
        return json.load(f)["eval_prompt"]


async def _score_all(rows, judges, max_concurrent):
    """rows: list of (row_idx, metric, question, answer); judges: {(trait, metric): OpenAiJudge}."""
    sem = asyncio.Semaphore(max_concurrent)

    async def one(i, key, q, a):
        async with sem:
            return i, key, await judges[key](question=q, answer=a)

    return await asyncio.gather(*[one(*r) for r in rows])


def judge_file(path, trait=None, judge_model="Qwen/Qwen2.5-14B-Instruct-AWQ", scoring="logprob",
               version="extract", prefix="", out_path=None, max_concurrent=32, metrics=("trait", "coherence")):
    """Score every row of a responses CSV with the local judge; return the scored DataFrame.

    trait:   trait name (e.g. "evil") for all rows, or None to use a per-row `trait` column.
    scoring: "logprob" (paper) or "text" (for judges that split numbers into digits, e.g. Qwen).
    version: which rubric file to use, data_generation/trait_data_{version}/{trait}.json.
    Scores are floats in [0, 100] or NaN (refusal / unparseable), as in the paper.
    """
    OpenAiJudge, Prompts = _paper_judge(scoring)
    df = pd.read_csv(path)
    traits = [trait] * len(df) if trait is not None else df["trait"].tolist()

    judges, rows = {}, []
    for i, (t, q, a) in enumerate(zip(traits, df["question"], df["answer"])):
        for m in metrics:
            key = (t, m)
            if key not in judges:
                template = _rubric(t, version) if m == "trait" else Prompts["coherence_0_100"]
                judges[key] = OpenAiJudge(judge_model, template, eval_type="0_100")
            rows.append((i, key, q, "" if pd.isna(a) else str(a)))

    results = asyncio.run(_score_all(rows, judges, max_concurrent))

    trait_col = f"{prefix}{trait}" if trait is not None else f"{prefix}trait"
    out = df.copy()
    out[trait_col] = float("nan")
    out[f"{prefix}coherence"] = float("nan")
    for i, (t, m), score in results:
        col = trait_col if m == "trait" else f"{prefix}coherence"
        out.loc[out.index[i], col] = float("nan") if score is None else score

    if out_path:
        out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--trait", default=None)
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    p.add_argument("--scoring", choices=["logprob", "text"], required=True)
    p.add_argument("--version", default="extract")
    p.add_argument("--prefix", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--max_concurrent", type=int, default=32)
    a = p.parse_args()
    out = judge_file(a.path, a.trait, a.judge_model, a.scoring, a.version, a.prefix, a.out, a.max_concurrent)
    print(out.filter(regex="coherence|trait|" + (a.trait or "")).describe().round(1))
