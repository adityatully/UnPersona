# unpersona

How do the trait directions from **Persona Vectors** (Chen et al. 2025, arXiv 2507.21509) work inside the
model? I'm asking which components write and read the evil / sycophantic / hallucinating directions, how a
system prompt creates a persona, and whether the directions are **necessary** as well as sufficient.

Model: Qwen2.5-7B-Instruct (bf16, 28 layers, d_model 3584). Main library: nnsight on the plain HF model.

## Status

| | what | state |
|---|---|---|
| setup 1–6 | paper pipeline, persona vectors, steering known-answer check (evil, layer 20, coef 2.0 → 96.1), graded prompts, helpers | done |
| [01](experiments/01_dla_prompt_last/README.md) | direct attribution of the `prompt_last` direction; fixed ruler; 3 traits; dilution | done (exploratory) |
| 02 | is the direction necessary? zero / mean ablation vs weight orthogonalization (later: negative steering) | being designed |

## Research log

The process, the questions and the dead ends are in `notes/`:
- [2026-11-23 — experiment 01 and the steering plan](notes/2026-11-23_log.md)

## Layout

```
experiments/NN_name/   README (question, method, expected, result, what I learned, open questions) + code + figures
notes/                 research log
unpersona/             helper loaders
patches/               small changes to the paper's code
third_party/           the paper's repo (not edited)
results/               small outputs; large artifacts live off-repo
```
