# CLAUDE.md — unpersona

This file tells Claude Code who I am, how I learn, what machine we run on, and how to work with me in this repo. Read it fully at the start of every session. For the setup tasks, also read `SETUP.md`.

## 1. Goal of this repo

- Paper: **Persona Vectors** (Chen et al., arXiv 2507.21509, code: https://github.com/safety-research/persona_vectors). It finds trait directions for **evil, sycophancy, hallucination** with a mean difference of activations, and shows that **adding** them causes the trait (sufficiency).
- My goal: understand **how the model works** in the paper's setting. How do system prompts create a persona inside the model? Which components write and read the persona directions? Which circuits produce the trait? I use the paper's data and vectors together with interp techniques: projection, patching, DLA, ablation vs negative steering, SAEs, transcoders. One of my questions is whether the directions are also **necessary**, not only sufficient.
- The paper uses two models: **Qwen2.5-7B-Instruct** (bf16, 28 layers, d_model = 3584; public SAEs exist) and **Llama-3.1-8B-Instruct** (public transcoders exist).
- I decide which experiment to run next, and I run and analyse it myself. Do not plan or implement experiments I have not asked for.

## 2. How I learn — follow this closely

- **Concrete artifacts first.** Explain every technique through the actual tensors: their names, shapes, and what each axis means. Example format:
  ```
  hidden_states[20]       [batch, pos, d_model] = [1, 57, 3584]   residual stream after decoder block 19
  o_proj.weight (layer 19) [d_model, n_heads*d_head] = [3584, 3584] maps all head outputs into the residual stream
  ```
  Always write shapes as comments next to tensors in code.
- **Deep reasoning, not recipes.** I want to know WHY a step is done, what would go wrong without it, and what the result means. Derive things from first principles (linear algebra view welcome: projections, subspaces, rank). Do not give me "just call this function".
- **Known answer first.** Each experiment should have a result we can check against. In this repo the main known answers are from the paper: steering evil at layer 20 (coef 2.0) gives a high trait score on Qwen (Figure 3); the projection of the last prompt token onto the persona vector rises with the system prompt's trait level (section 3.3). If my result differs, we find out why.
- **Tweak internals.** After a technique works, help me poke at it: change a layer, ablate a head, swap a metric, break an assumption, and predict the result BEFORE running it.
- **Linear algebra is my main lens.** I think in terms of the four fundamental subspaces (Strang), rank-nullity, SVD, eigenvalues, "matrix × column = combination of columns, row × matrix = combination of rows". Use this language when it helps.
- **State the convention every time.** My notes use one token as a column vector and weights in `[out, in]` form (output = W x). HF stores `nn.Linear.weight` as `[out, in]` too (same as my notes). TransformerLens stores weights as `[d_model, d_head]` and tokens as rows, so things look transposed. When a formula and code disagree in shape, say which convention each one uses.
- **How I learn best:** I write my own understanding (often long, with my own shapes and examples) and ask you to check it. Then:
  - Say exactly which part is right and which part is wrong, and why. Do not rewrite my whole reasoning.
  - Use tiny numeric examples (a 3-dim toy residual stream, a 2-feature table) to make a point concrete.
  - Mark the level of certainty: **(1) mathematical fact**, **(2) strong empirical evidence**, **(3) open hypothesis / debated**. Example: "zero-ablation sets v̂ᵀh to 0" is (1); "the persona vector is the direction the model uses for the trait" is (3). Do not let me reason from a level-3 claim as if it were a fact.
- Language: plain, clear English (short sentences, active voice). Keep needed detail, skip filler and jargon that adds nothing.

### What I already understand (do not re-teach from zero; build on it)

From my notes (Notion → "mech interpretability"):
- Toy models of superposition: features as directions, interference = dot product, why ReLU + negative bias remove interference, sparsity decides how many features fit, importance decides which stay, privileged vs non-privileged basis.
- Residual stream as the communication channel (no privileged basis); heads read/write through low-rank maps; attention heads are independent and additive.
- Attention head as `A ⊗ W_OV`: QK circuit = where to move information from, OV circuit = what is moved. Full QK / OV circuits, Q/K/V composition, virtual weights, induction heads.
- Four-subspace reading of W_OV: col = where it writes, row = what it reads, null = what it ignores, left null = what it never changes.
- MLPs: W_in rows = detectors, W_out columns = writers; "computation" = nonlinear interaction; the residual connection keeps inputs.
- SAEs: an overcomplete dictionary (not a basis), F is a hyperparameter, feature splitting, no guarantee the SAE finds the model's real features. Probes often beat SAEs on downstream tasks.
- Techniques I derived myself (in my "circuit analysis" doc): DLA, logit lens, activation patching, path patching, composition scores, linear probes, probe vs mean-difference directions, steering vectors, directional ablation, attribution graphs (basics).
- Pre-norm transformers: blocks read a normalized copy of the residual stream; the stream itself is a plain sum and its norm grows with depth.

## 3. Working rules

- **Disagree with me when I am wrong. Do not just agree.**
- **Flag common traps:** BOS token, off-by-one layer indices (`hidden_states[ℓ]` = output of block ℓ−1), prompts with different token lengths in patching, chat template mistakes (system prompt, `add_generation_prompt`), bf16 precision in metrics, the judge's randomness.

## 4. Workstation: coe9 (shared, remote via SSH)

| Part | Details |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8 |
| CPU | 2 × Xeon Silver 4210R (2 sockets × 10 cores = 20 cores / 40 threads). **Shared, often fully used.** PCIe 3.0 only. |
| RAM | 251 GB (~233 GB free), 2 GB swap |
| GPU | 1 × RTX A5000 (Ampere), 24 GB VRAM, ~768 GB/s. bf16 yes, **no native FP8**. Shared but usually free. Driver 580. |
| Root disk `/` | 1 TB NVMe, **~11 GB free. Never put large files here.** |
| `/data` | 3.6 TB RAID (ext4), ~229 MB/s read. **My folder: `/data/coe9`. Probably no backup.** |

Shared-machine rules:
- Run `nvidia-smi` before big GPU jobs. Do not take the GPU for long without reason.
- Keep CPU threads low (`OMP_NUM_THREADS=4`). Pin to the GPU's socket if needed (`nvidia-smi topo -m`, then `taskset`).
- Long jobs run inside `tmux`.

## 5. Environment and paths

- Conda envs on `/data/coe9/envs` (`interp` and `gen` already exist from my learning repo):
  - `interp`: torch, nnsight, transformers, accelerate, transformer_lens, sae-lens, bitsandbytes, scikit-learn, jupyter. **My experiments run here.**
  - `gen`: vLLM only (it pins its own torch version; never install it into `interp`). Used for the local judge server.
  - `persona` (new): the paper's own pipeline (its `requirements.txt`).
  - `circuits` (new, phase 2): circuit-tracer fork for Llama transcoders.
- Env vars (in `~/.bashrc`): `HF_HOME=/data/coe9/hf_cache`, `PIP_CACHE_DIR`, `TORCH_HOME`, `TMPDIR` all on `/data/coe9`; `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `TOKENIZERS_PARALLELISM=false`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Large artifacts of this repo (responses, vectors, activations, judge outputs): `/data/coe9/artifacts/unpersona/` (gitignored).
- Gated models (Llama) need `hf auth login` (I do this myself).
- I connect with VS Code / Cursor Remote-SSH and use notebooks with the `interp (coe9)` kernel.

## 6. Library and model rules

- **Main library: nnsight on the plain HF model.** The paper's vectors come from HF `hidden_states`, and nnsight hooks the same modules, so the vectors match exactly. Load with `torch_dtype=torch.bfloat16`, `attn_implementation="eager"` (needed for attention patterns).
- TransformerLens only for cross-checks, and only with `from_pretrained_no_processing` (the default weight processing changes the activations, so the paper's vectors would not match). Verify once that its residual stream equals HF `hidden_states`.
- Per-head outputs in HF/nnsight: take the input of `self_attn.o_proj`, `[batch, pos, 3584]`, reshape to `[batch, pos, 28, 128]`; head h writes `o_proj.weight[:, 128h:128(h+1)] @ z_h` (`[3584, 128] @ [128]`). Qwen2.5-7B has 28 query heads and 4 KV heads (grouped-query attention).

## 7. Memory and precision rules

- VRAM budget: weights + CUDA context (~0.5 GB) + activations + KV cache + cached activations must fit in 24 GB.
- Weights: params × bytes (bf16 = 2, int8 = 1, nf4 ≈ 0.55). 7–8B bf16 ≈ 15–16.4 GB (fits). 14B bf16 ≈ 28 GB (does NOT fit). The 7B model and the 14B judge do not fit together: generate and save first, then judge.
- Always compute metrics (projections, trait-score differences, patching scores, KL) in fp32: `.float()`.
- Do not cache all activations of long generations. Save only the layers and positions needed.
- Always use `tokenizer.apply_chat_template`; the system prompt is part of the experiment.
- Saved activation sets go in `/data/coe9/artifacts/unpersona/`, never `/`.

## 8. Repo layout and conventions

```
CLAUDE.md                        this file
SETUP.md                         setup spec (setup tasks only)
third_party/persona_vectors/     the paper's repo (do not edit in place)
patches/                         small changes to the paper's code (local judge, paths)
unpersona/                  helper loaders (boilerplate)
experiments/                     one folder per experiment, written by me
  01_<name>/
    README.md                    question, metric, expected result, actual result, what I learned
    main.ipynb  (or .py)
notebooks/                       scratch
results/                         small outputs (plots, json). Large files go to /data/coe9/artifacts/unpersona (gitignored)
notes/                           short summaries; longer notes live in my Notion
```

- Each experiment README ends with "What I learned" and "Open questions" written by me (you may draft from our discussion, I edit).
- Fixed seeds; record model name, dtype, judge model, library versions in each README.
- Git: commit small and often; push to GitHub (the machine may have no backup). Never commit weights, caches or large activation files.

## 9. Notion

- My notes are in Notion under the page "mech interpretability" (sub-pages: Core techniques, tranformer, MLP's, SAE AND DICTIONARY LEARNING, …).
- If the Notion MCP server is connected, read the relevant page before explaining a topic, so you build on what I already wrote and use my notation.
- Do not edit or create Notion pages unless I ask. If I ask for an experiment summary in Notion, put it as a new sub-page under "mech interpretability".
