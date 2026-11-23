# 03 — Mean ablation for sycophancy and hallucination

## Question
In experiment 02b, mean ablation along v̂_20 (the standard evil direction) removed almost all evil behaviour
(76 → 5), with coherence intact. Does the same hold for the other two traits? Is each trait's own
direction necessary for that trait?

## Method
The same as experiment 02b (`experiments/02_zero_ablation/mean_ablation.py`), applied to another trait.

| trait | direction (one fixed vector at every layer and token) | why this layer |
|---|---|---|
| sycophancy | v̂_20 (the standard sycophancy direction) = `sycophantic_response_avg[20]` | the paper's steering peak |
| hallucination | v̂_16 (the standard hallucination direction) = `hallucinating_response_avg[16]` | the paper's steering peak |

- Mean ablation: `h' = h − (v̂ᵀh − μ_l) v̂` at every stream point l = 0..28 and every token. There are separate
  normal values for prompt and answer tokens, taken from the no-prompt run. The first 3 tokens stay
  untouched.
- Per trait: the trait's 20 eval questions, greedy decoding, max 1000 new tokens.

  | condition | n | purpose |
  |---|---|---|
  | trait_base: 5 trait system prompts, no ablation | 100 | baseline: should show the trait |
  | trait_mean: same, mean ablation | 100 | the necessity test |
  | noprompt_base: Qwen default system prompt | 20 | source of μ |
  | noprompt_mean: same, mean ablation | 20 | known answer: unchanged. Hallucination's normal level is not 0 (the paper shows a 15–20 baseline). |
- Judge: Codex (gpt-5.6-sol), each trait's own paper rubric plus coherence, with hidden anchors.

Run (env interp, on coe9):
```bash
python experiments/03_mean_ablation_traits/run.py sycophantic 20
python experiments/03_mean_ablation_traits/run.py hallucinating 16
python experiments/03_mean_ablation_traits/analyze.py
```

## Expected result (written before running)
_(my prediction)_

## Checks
_(after the run)_

## Result
_(after the run)_

## What I learned
_(to be written by me)_

## Open questions
- Specificity: mean-ablate v̂_20 (the standard evil direction) and give the model the sycophantic prompts.
  Does sycophancy survive? (planned next)
