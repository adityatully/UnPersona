# 04 — Specificity: does the evil direction also carry sycophancy?

## Question
Mean ablation along v̂_20 (the standard evil direction) removed evil (76 → 5). Is that direction specific
to evil, or does it also carry something general that other personas need (for example "follow the system
prompt's persona")? Test: sycophantic system prompts, with mean ablation along the evil direction.

## Method
- Directions (unit length, one fixed vector at every layer and token):
  - `e` = v̂_20 (the standard evil direction)
  - `s` = v̂_20 (the standard sycophancy direction), with cos(e, s) = 0.369
  - `e⊥ = (e − (eᵀs) s) / ‖·‖`: the evil direction with its sycophancy part removed. cos(e⊥, s) = 0 (1);
    cos(e⊥, e) = 0.93.
- Mean ablation as in experiments 02b and 03. μ along each direction comes from sycophancy's own no-prompt
  answers (experiment 03).
- Conditions (sycophancy's 20 eval questions, greedy):

  | condition | n | purpose |
  |---|---|---|
  | sycophantic prompt + mean ablation along `e` | 100 | raw specificity |
  | sycophantic prompt + mean ablation along `e⊥` | 100 | overlap control |
  | no prompt + each ablation | 20 + 20 | known answers: unchanged |
  | reused from 03: sycophantic prompt, no ablation / ablation along `s` | 100 / 100 | 68.5 / 14.4 |
- Judge: Codex, sycophancy rubric plus coherence.

## Reading the result
| result | meaning |
|---|---|
| both ≈ 68.5 | the evil direction is evil-specific (relative to sycophancy) |
| `e` drops it, `e⊥` doesn't | the drop came from geometric overlap |
| both drop it a lot | shared content (e.g. persona following) |

Noise: the spread is about ±32, so a difference between two means of 100 answers is uncertain by about ±5.
Treat differences under ~10 as noise.

## Expected result (written before running)
_(my prediction)_

## Result
_(after the run)_

## What I learned
_(to be written by me)_
