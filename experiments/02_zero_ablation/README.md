# 02 — Is the evil direction necessary? All-layer zero ablation

## Question
Steering with the evil `response_avg` vector at layer 20 makes the model evil (sufficiency; task 4: evil 96.1
at coef 2.0). If we **remove** that direction everywhere, does an evil-prompted model stop being evil?

## Method
- Direction: `v̂ = v / ‖v‖`, `v = evil_response_avg_diff[20]` `[3584]`, fp32. One fixed `v̂` at every layer
  and every position.
- Zero ablation: no component may write along `v̂`. Each writer's output `o` `[batch, pos, 3584]` becomes
  `o − (o·v̂) v̂`. The writers are `embed_tokens`, 28 × `o_proj`, and 28 × `mlp.down_proj`. The stream is a sum
  of writes, so the dial `v̂ᵀh_l` is 0 at every layer and token (1). It applies to prompt tokens (prefill) and
  to every generated token.
- Check, done a second way: weight orthogonalization. `W ← (I − v̂v̂ᵀ) W` for `o_proj` / `down_proj`
  (`[out, in]`), and `E ← E (I − v̂v̂ᵀ)` for the embedding (one token per row). In exact maths this is the
  same model (1). In bf16, compare next-token logits rather than text.
- Conditions: greedy decoding, max 1000 new tokens (the paper's limit), and Qwen's
  `repetition_penalty = 1.05` from `generation_config`, kept.

  | # | system prompt | intervention | n |
  |---|---|---|---|
  | 1 evil_base | 5 evil prompts (from extraction) × 20 eval questions | none | 100 |
  | 2 evil_ablate | same | hooks | 100 |
  | 3 evil_ortho | same | orthogonalized weights | 100 |
  | 4 noprompt_base | Qwen default ("You are Qwen, created by Alibaba Cloud…") × 20 | none | 20 |
  | 5 noprompt_ablate | same | hooks | 20 |
- Metric: the same judge as task 4. That is the Codex judge (`unpersona/codex_judge.py`, gpt-5.6-sol, paper
  rubric, hidden anchors), scoring evil 0–100 and coherence 0–100. Also: read ~10 answers per condition.
- Internals recorded per answer: dial `v̂ᵀh_l` (l = 0..27), averaged over prompt tokens (position 0 left
  out) and over answer tokens. Also the logits of the first answer token.
- Model Qwen/Qwen2.5-7B-Instruct, bf16, eager attention, batch 20, left padding. Versions are in
  `meta_*.json` in the artifacts folder.

Run (env interp, repo root on coe9):
```bash
python experiments/02_zero_ablation/generate.py hooks
python experiments/02_zero_ablation/generate.py ortho
python -m unpersona.codex_judge build ...   # then patches/run_codex_scoring.sh on the Mac, then collect
python experiments/02_zero_ablation/analyze.py
```

## Expected result (written before running)
_(my prediction)_

Known answers: condition 1 high evil; condition 4 ≈ task 4 baseline (evil 0.0, coherence 98.7); conditions
2 and 3 nearly identical.

## Checks
_(after the run)_

## Result
_(after the run)_

## What I learned
_(to be written by me)_

## Open questions
_(to be written by me)_
