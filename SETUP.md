# SETUP.md — unpersona

Setup spec for Claude Code. Read this together with `CLAUDE.md` (my general rules: tutor mode, coe9 machine, memory and precision rules). If the two disagree, `CLAUDE.md` wins.

## What this repo is for (context only)

- Paper: **Persona Vectors** (Chen et al., arXiv 2507.21509). Code: https://github.com/safety-research/persona_vectors
- The paper finds trait directions ("persona vectors") for **evil, sycophancy, hallucination** by mean difference of activations (trait-prompted responses − non-trait responses). It shows that **adding** the vector causes the trait.
- My goal: understand **how the model works** in the paper's setting. How do system prompts create a persona inside the model? Which components write and read the persona directions? Which circuits produce the trait? I use the paper's data and vectors together with interp techniques (projection, patching, DLA, ablation, SAEs, transcoders). One of my questions is whether the directions are also **necessary**, not only sufficient.
- The paper uses two models: **Qwen2.5-7B-Instruct** (bf16, 28 layers, d_model = 3584) and **Llama-3.1-8B-Instruct**. Public SAEs exist for the Qwen model; public transcoders exist for the Llama model.

**Your job is setup only.** Get the data, the vectors, the judge, the loaders and the sanity checks working. Do **not** write experiment code (steering hooks, ablation, patching, DLA, probes, circuit analysis). When setup is done, stop and give me a short report.

## Machine and storage

- coe9, RTX A5000 24 GB, follow `CLAUDE.md` section 4–6.
- Everything large goes in `/data/coe9/artifacts/unpersona/` (gitignored): generated responses, vectors, cached activations, judge outputs.
- Model weights go to `HF_HOME=/data/coe9/hf_cache` (already set).
- Run `nvidia-smi` before GPU jobs. Long jobs in `tmux`.

## Repo layout

```
CLAUDE.md                 repo-specific rules (already written)
SETUP.md                  this file
third_party/persona_vectors/   the paper's repo (git clone, do not edit in place; put patches in patches/)
patches/                  my small changes to the paper's code (local judge, paths)
unpersona/           my helper package (loaders only, see below)
experiments/              empty; I fill it
notebooks/                empty; I fill it
results/                  small outputs only
```

## Environments

Use the existing conda envs on `/data/coe9/envs`. Do not install vLLM into `interp`.

| Env | Use |
|---|---|
| `persona` (new) | Run the paper's own pipeline (`requirements.txt` of the paper repo). Check whether it pins torch / vLLM; if it conflicts with `gen`, keep it separate. |
| `gen` (exists) | vLLM server for the local judge. |
| `interp` (exists) | My experiments: **nnsight** + transformers (main), TransformerLens only for cross-checks. |
| `circuits` (new, phase 2 only) | circuit-tracer fork for Llama transcoders. |

**Library choice: nnsight on the plain HF model.** Reason: the paper takes vectors from HF `hidden_states`. nnsight hooks the same HF modules, so there is no conversion mismatch. It also supports interventions during `generate`, which I need for steering and ablation. Load with `attn_implementation="eager"` and `torch_dtype=torch.bfloat16`.

## Tasks, in order

### 1. Clone and install the paper repo
- Clone into `third_party/persona_vectors`, create env `persona`, `pip install -r requirements.txt`, `unzip dataset.zip` if needed.
- Report: Python version, torch version, whether generation uses vLLM or HF, and the list of traits in `data_generation/trait_data_extract/` and `trait_data_eval/`.

### 2. Local judge (I have no OpenAI or Anthropic API credits)
- The paper uses `gpt-4.1-mini` as judge through the OpenAI client (code adapted from the Emergent Misalignment repo).
- Read the judge code first and tell me **how the score is computed** (plain text number, or a weighted average over the top logprobs of number tokens).
- Serve a local judge with vLLM's OpenAI-compatible server in env `gen`. Default: `Qwen/Qwen2.5-14B-Instruct-AWQ`. Make the paper's code use it through `OPENAI_BASE_URL` / `base_url`, with as small a patch as possible (in `patches/`). If the scoring uses logprobs, check that vLLM returns enough `top_logprobs`.
- VRAM: the judge and the 7B model do **not** fit together. Run in stages: generate and save responses → stop the model → start the judge and score the saved files.
- Calibration (boilerplate): write a script that samples 30 responses per trait, both high and low scores, into a CSV with an empty column `my_score`. I will score them by hand. Then a script that reports agreement between my scores and the judge.
- Optional second judge, only when I ask: you (Claude Code) score the same calibration CSV with the paper's rubric, writing a plain number per row into a column `claude_score`, without looking at the other score columns. Use this only on small files (≤ 100 rows), never for bulk judging.

### 3. Generate the persona vectors
- For each trait (evil, sycophantic, hallucinating): run extraction with pos and neg system prompts, judge the responses, then `generate_vec.py`. Model: `Qwen/Qwen2.5-7B-Instruct`.
- Save to `/data/coe9/artifacts/unpersona/vectors/Qwen2.5-7B-Instruct/`. Keep all three variants (`response_avg_diff`, `prompt_avg_diff`, `prompt_last_diff`).
- Report for each trait: number of responses kept after filtering, vector shape, and norm per layer.
- **Check the layer index convention.** The vectors are stored as `[n_layers × d_model]`. Find out if row 0 is the embedding output (HF `hidden_states[0]`) and so row ℓ = output of decoder block ℓ−1. Write the answer in the report, with the code line that proves it.

### 4. One known-answer check
- Run the paper's own steering eval: evil, layer 20, coef 2.0, `steering_type response`, plus the unsteered baseline, on the eval questions.
- Expected (paper Figure 3, Qwen): the steered evil score is high (roughly 90), the baseline is near 0. Our judge is different, so I expect the same direction, not the same numbers. Report both numbers.
- Do not run any other steering sweeps.

### 5. Graded system prompts
- The paper has a figure (section 3.3) with system prompts at 8 levels of trait encouragement (index 0–7) for each trait. Find these prompts in the repo or the paper's appendix and save them as `data/graded_prompts.json` (`{trait: [8 prompts]}`).
- If you cannot find them, tell me. Do not write your own.

### 6. Helper package `unpersona/` (boilerplate only)
Write these small, readable functions, with shape comments on every tensor (my convention: see `CLAUDE.md`):
- `load_model()` → nnsight `LanguageModel`, bf16, eager attention.
- `load_vectors(trait, variant)` → tensor `[n_layers, d_model]`, plus a docstring that states the layer index convention from task 3.
- `build_prompt(system, user)` → chat-templated string via `tokenizer.apply_chat_template(..., add_generation_prompt=True)`; and `last_prompt_pos(...)` → index of the last prompt token.
- `load_questions(trait, split)` → extraction / eval questions from the paper's data.
- `load_mmlu_subset(n=500, seed=0)` → fixed MMLU subset, and a plain (no-intervention) accuracy function for the base model.
- `judge_file(path)` → scores a saved responses file with the local judge.
- `load_sae(layer)` → see task 7.

### 7. SAEs (Qwen2.5-7B-Instruct)
- Public SAEs from a paper co-author: `andyrdt/saes-qwen2.5-7b-instruct` on the HF Hub (131,072 features, BatchTopK, every 4 layers; also on Neuronpedia as `qwen2.5-7b-it`, "Resid Post - 131k").
- List the available layers and trainers. Find the trainer with **k = 64** (the paper uses k = 64). Report which hook each SAE reads (`resid_post` of which block) and how that maps to the paper's layer numbers (task 3).
- Write `load_sae(layer)` returning `W_enc`, `b_enc`, `W_dec`, `b_dec`, and the inference sparsity rule (BatchTopK usually uses a global threshold at inference; use what the SAE's own code uses).
- Sanity check on ~50 real chat prompts: fraction of variance explained and mean L0 (should be near 64). Report both.

### 8. Sanity checks (report results, do not skip)
- nnsight residual at decoder block ℓ output equals HF `hidden_states[ℓ+1]` (max abs diff, in fp32).
- Projection check: mean of `v̂_20ᵀ h` over pos-prompt responses is clearly larger than over neg-prompt responses, for each trait.
- Determinism: same seed → same generations with greedy decoding.

### 9. Phase 2 (only when I ask): Llama-3.1-8B-Instruct + transcoders
- Gated model: I run `hf auth login` myself.
- Transcoders: `facebook/crv-8b-instruct-transcoders` (TopK transcoders). They need the circuit-tracer fork at https://github.com/zsquaredz/circuit-tracer (upstream does not support TopK yet). Install in env `circuits` and run only the fork's minimal example on one short prompt.
- Generate persona vectors for Llama with the same pipeline as task 3.

## Final report (when you stop)
- What works, what failed, and the numbers from tasks 3, 4, 7, 8.
- Exact versions: torch, transformers, nnsight, vLLM, judge model.
- Disk used on `/data/coe9/artifacts/unpersona/`.
- Any place where you had to guess. Mark it clearly.
