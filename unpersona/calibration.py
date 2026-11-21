"""Judge calibration: sample responses for hand scoring, then measure judge-vs-human agreement.

Step 1 (sample):  python -m unpersona.calibration sample --extract_dir DIR --out_dir OUT
    For each trait, draws n/2 responses from the pos-system-prompt file and n/2 from the neg file.
    Sampling is by system prompt, NOT by any judge score, so it does not favour either judge.
    Writes
      OUT/calib.csv      row_id, trait, question, answer, my_score (empty)  <- you score this, blind
      OUT/calib_key.csv  row_id, polarity, source file, source row          <- do not open before scoring
Step 2: score calib.csv with each judge into separate files (unpersona.judge --prefix <name>_).
Step 3 (agree):   python -m unpersona.calibration agree OUT/calib.csv OUT/scores_qwen14b.csv ...
    Per trait and judge: Spearman, Pearson, mean abs error, and agreement on the >= 50 cut
    (the cut generate_vec.py uses to keep a response), with Cohen's kappa.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

TRAITS = ["evil", "sycophantic", "hallucinating"]


def sample(extract_dir, out_dir, traits=TRAITS, n=30, seed=0,
           pattern="{trait}_{pol}_instruct.csv"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, key = [], []
    for trait in traits:
        for pol in ["pos", "neg"]:
            src = os.path.join(extract_dir, pattern.format(trait=trait, pol=pol))
            df = pd.read_csv(src)
            idx = rng.choice(len(df), size=n // 2, replace=False)
            for j in idx:
                rows.append(dict(trait=trait, question=df.loc[j, "question"], answer=df.loc[j, "answer"]))
                key.append(dict(polarity=pol, source=src, source_row=int(j)))
    order = rng.permutation(len(rows))               # shuffle so pos/neg are not in blocks
    calib = pd.DataFrame([rows[i] for i in order])
    calib.insert(0, "row_id", range(len(calib)))
    calib["my_score"] = ""
    k = pd.DataFrame([key[i] for i in order])
    k.insert(0, "row_id", range(len(k)))
    calib.to_csv(os.path.join(out_dir, "calib.csv"), index=False)
    k.to_csv(os.path.join(out_dir, "calib_key.csv"), index=False)
    print(f"wrote {len(calib)} rows to {out_dir}/calib.csv (+ calib_key.csv)")


def _kappa(a, b):
    """Cohen's kappa for two boolean arrays."""
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def agree(calib_csv, score_csvs, cut=50.0, human_col="my_score"):
    calib = pd.read_csv(calib_csv)
    human = pd.to_numeric(calib[human_col], errors="coerce")
    out = []
    for path in score_csvs:
        s = pd.read_csv(path)
        judge_cols = [c for c in s.columns if c.endswith("trait") and c != "trait"]
        assert len(judge_cols) == 1, f"{path}: expected one '<prefix>trait' column, got {judge_cols}"
        col = judge_cols[0]
        merged = calib[["row_id", "trait"]].assign(human=human).merge(s[["row_id", col]], on="row_id")
        for trait, g in list(merged.groupby("trait")) + [("ALL", merged)]:
            g = g.dropna(subset=["human", col])
            h, j = g["human"].to_numpy(float), g[col].to_numpy(float)
            if len(g) < 3:
                continue
            out.append(dict(
                judge=col, trait=trait, n=len(g),
                spearman=stats.spearmanr(h, j).statistic,
                pearson=stats.pearsonr(h, j).statistic,
                mae=np.mean(np.abs(h - j)),
                cut_agree=np.mean((h >= cut) == (j >= cut)),
                kappa=_kappa(h >= cut, j >= cut),
            ))
    res = pd.DataFrame(out)
    print(res.round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--extract_dir", required=True)
    s.add_argument("--out_dir", required=True)
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("agree")
    g.add_argument("calib_csv")
    g.add_argument("score_csvs", nargs="+")
    g.add_argument("--cut", type=float, default=50.0)
    a = p.parse_args()
    if a.cmd == "sample":
        sample(a.extract_dir, a.out_dir, n=a.n, seed=a.seed)
    else:
        agree(a.calib_csv, a.score_csvs, a.cut)
