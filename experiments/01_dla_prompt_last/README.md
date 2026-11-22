# 01 — Direct attribution of the evil prompt_last direction

## Question
At the last prompt token (`\n` after `<|im_start|>assistant`), which components (embedding, 28 attention
blocks split into 784 heads, 28 MLPs) directly write the difference between evil-prompted and
helpful-prompted runs along the evil `prompt_last` persona direction?

## Method
- Readout direction `v̂_L = v_L / ‖v_L‖`, `v_L = evil_prompt_last_diff[L]`, for L = 20 (paper's steering
  layer) and L = 28 (last).
- Exact identity at one position (residual stream is a plain sum; level 1):
  `hidden_states[L] = embed + Σ_{l<L} (attn_l + mlp_l)`, so `v̂ᵀ h_L = Σ_c v̂ᵀ out_c`.
  Heads: `attn_l = Σ_h W_O[:, 128h:128h+128] z_{l,h}` (o_proj has no bias); `v̂ᵀ(W_O,h z_h) = u_{l,h}ᵀ z_h`,
  `u_{l,h} = W_O,hᵀ v̂` [128].
- **L = 28 is post-final-norm**: HF's `hidden_states[28]` = `model.norm(block 27 output)` (verified). The norm
  `γ ⊙ x · s`, `s = 1/rms(x)`, is made linear by freezing `s` from each prompt's own forward pass; readout
  becomes `w = s · γ ⊙ v̂`. No norm is involved for L = 20 (raw residual stream).
- Attributed quantity: `Δ_c = v̂ᵀ(mean_evil out_c − mean_helpful out_c)`; `Σ_c Δ_c = v̂ᵀ(mean_evil h_L − mean_helpful h_L)`.
- Data: **extract** = prompts of the 884 kept evil pairs (100 unique per side, weighted by count) — these
  built `v`, so the attribution is an exact decomposition of the vector; **eval** = the same 5 evil / 5 helpful
  system prompts × 20 eval questions.
- Model Qwen/Qwen2.5-7B-Instruct, bf16, eager attention, nnsight 0.8.2, transformers 5.23.0, torch 2.15.0;
  projections in fp32; batch size 1; no sampling (deterministic).

## Expected result (written before running)
Nothing from MLP 0 (identical token and embedding at this position); attention in early-mid layers imports
the difference; later MLPs carry a large share; contributions grow with depth; a handful of heads stand out.

## Checks
- Vector reconstruction (known answer): `mean_pos − mean_neg` of our hooked `hidden_states[L]` vs the saved
  vector: L20 cos 0.999, norm ratio 0.99; L28 (after applying the final norm) cos 0.999, norm ratio 0.99;
  layers 1–10 cos ≥ 0.96 (small vectors, norm 2–8; likely bf16 eager-vs-SDPA rounding — not checked).
- Sum of components = readout difference to 3 decimals (48.883 vs 48.885 at L20; 144.643 vs 144.641 at L28);
  sum of heads = attention total (|diff| ≤ 2e-3).

## Result
| | readout Δ | attention | MLP | top components |
|---|---|---|---|---|
| extract, L20 | 48.8 | 13.4 (27%) | 35.4 (73%) | MLP19 13.3, MLP18 8.8, ATTN19 6.2, MLP13 3.1, MLP17 2.9 |
| eval, L20 | 48.9 | 13.0 | 35.9 | MLP19 13.1, MLP18 9.1, ATTN19 5.8 |
| extract, L28 | 141.8 | 18.8 (13%) | 123.1 (87%) | MLP27 57.3, MLP26 19.9, MLP25 13.5, MLP24 8.7 |
| eval, L28 | 144.6 | 19.0 | 125.6 | MLP27 60.6, MLP26 20.4 |

Top heads at L20: L19H27 +2.9, L19H0 −1.8 (writes against), L19H22 +1.6, L19H2 +1.5, L19H26 −1.3, L19H4 +1.1,
L14H16 +1.0. Figures: `fig1_layer_contributions.png`, `fig2_heads_L20.png`, `fig3_buildup.png`.

Context needed to read this (from the same vectors):
- The direction rotates with depth: cos(v_l, v_20) = 0.12 (l=5), 0.34 (14), 0.62 (18), 0.79 (19);
  cos(v_l, v_27) = 0.40 (20), 0.72 (24), 0.90 (26).
- Partial-sum readout `v̂_20ᵀ h_l`, evil − helpful: 1.1 (l=10), 6.9 (14), 19.4 (18), 29.3 (19), 48.8 (20).
- Fig 3: relative to the projection's zero, the helpful prompt moves far negative (−33 at L20) while the
  evil prompt moves less positive (+16). The zero point is arbitrary (no neutral baseline).


## Follow-up: a fixed ruler (figs 4–6)

**Why.** Figs 1–2 used a ruler (`v̂_L`) built at the readout layer L. Changing L from 20 to 28 moved the top
writers from MLP 18/19 to MLP 26/27. Was that something in the model, or something in the ruler? To separate
the two, I kept one ruler fixed, the evil `v̂_20`, and scored **every** block 0–27 against it. This includes
blocks 20–27, which come after the layer the ruler was built at.

**What one bar is.** For block l: `v̂_20ᵀ(mean over evil-prompt runs of out_l − mean over helpful-prompt runs of out_l)`,
at the last prompt token, extract prompts. Subtracting removes what the block writes in both runs. Toy:
block writes 15 along `v̂_20` in the evil run and 12 in the helpful run, so 3 units are caused by the evil prompt.

| figure | script | what it shows |
|---|---|---|
| `fig4_fixed_ruler_v20_all_blocks.png` | `fixed_ruler.py` | evil: all 28 blocks against `v̂_20`, plus the running total |
| `fig5_fixed_ruler_v20_3traits.png` | `fixed_ruler_traits.py` | the same bar chart for evil, sycophantic, hallucinating (each against its own `v̂_20`) |
| `fig6_share_vs_raw_3traits.png` | `share_vs_raw.py` | per layer: raw amount `v̂_20ᵀh_l` and share `v̂_20ᵀh_l / ‖h_l‖`, trait run and helpful run separately |

Other traits: `python main.py sycophantic` / `hallucinating` (artifacts in `exp01_dla_prompt_last/<trait>/`;
reconstruction checks in `check_vector_reconstruction_<trait>.json`).

**Checks.** The sum of all components equals the running total at the end. Evil extract: 56.155 vs 56.130.
Eval: 53.302 vs 53.309.

**Results.**
- Blocks 0–19 build the amount along `v̂_20` (48.8 at layer 20). Blocks 20–26 add more, but with small
  bars. Block 27 writes **against** it: evil MLP27 ≈ −20.6 (attention 27 ≈ +3).
- Same shape in all three traits; the last block's push-back is strongest for evil.

  | trait | built along `v̂_20` before block 27 | block 27 | fraction removed |
  |---|---|---|---|
  | evil | ≈ 77 | ≈ −21 (MLP) | ≈ 25% |
  | sycophantic | ≈ 83 | ≈ −14 (attn ≈ −4.6, MLP ≈ −9) | ≈ 17% |
  | hallucinating | ≈ 56 | ≈ −3 | ≈ 6% |

  Hallucinating also has small negative MLPs around 20–22.
- The total stays positive after block 27 (evil: ≈ 77 → 56). Block 27 **reduces** the persona along `v̂_20`;
  it does not reverse it.
- Fig 6: the raw gap (trait − helpful) keeps growing after layer 20 (evil 48.8 → 73.7 at layer 27;
  sycophantic 53.3 → 83.1; hallucinating 47.5 → 56.5). The **share** peaks around layer 20 in the trait run and
  then falls toward 0, because the rest of the stream grows faster (dilution).

## What I learned
_(draft from our discussion — to be edited by me)_

Certainty: (1) mathematical fact, (2) strong empirical evidence, (3) open hypothesis.

1. **DLA works as an exact bookkeeping tool here.** The residual stream is a plain sum, so the per-component
   scores add up to the total (checked to 2–3 decimals). (1)
2. **Subtracting the two runs isolates what the system prompt caused.** Without it, a component's score
   includes what it always writes at this token (e.g. "an answer starts here"). (1)
3. **DLA onto `v_L` always favours the blocks just before L, because the ruler is built from their writes.**
   `v_L` is the sum of all writes before L. The latest large writes dominate that sum, so the latest writers
   always look the most aligned. The ranking says as much about the ruler as about the model. (1)
4. **The direction rotates with depth.** cos(v_20, v_27) = 0.40, so only ≈ 16% of `v_27`'s squared length
   lies along `v̂_20`. Rotation comes from vector addition (new writes in new directions), not from anything
   being deleted. Nothing is ever removed from the stream unless a component writes a negative amount. (1)
5. **A fixed ruler shows what later blocks do to an earlier direction.** Blocks 20–26 add a little; block 27
   pushes back, in all three traits. (2) on these prompts.
6. **Raw amount ≠ share.** Each block reads a normalized copy of the stream, so what it "notices" is roughly
   the share `v̂ᵀh / ‖h‖`. The persona's share peaks around layer 20 and then gets diluted. This may be part of
   why the paper's steering works best at layer 20. The pattern is (2); the link to steering is (3).
7. **Dilution and a negative write are different things.** Dividing by a positive norm can only make the
   persona quieter, never flip its sign (1). MLP27's write is an actual negative push.
8. **Writing ≠ reading.** DLA only shows who writes along a direction. It never shows who uses it, or whether
   it is necessary. (1)
9. **0 is not a neutral point along the persona direction.** At layer 20 the helpful run sits at ≈ −32 and the
   evil run at ≈ +16. Any method that sets this value to 0 (zero-ablation) pushes helpful runs toward evil. (1)

## Open questions
_(draft — to be edited by me)_

- **What does MLP27 do?** Three stories, all (3): a clean-up before the output; a general shrink /
  confidence regulation (it also writes against `v̂_20` in the helpful run alone); or conversion (it reads
  evil and writes its consequence, e.g. a push toward an evil first token, in a direction partly opposite to
  `v̂_20`). Known mechanisms of this type in other models: Hydra effect / erasure, copy suppression, entropy /
  confidence neurons. These references were cited from memory and have not been checked. Parked.
- **What do the new late directions mean?** (A) the same "evil", re-encoded; or (B) downstream things caused
  by evil, like an answer plan or the first token "Let". Possible tests: logit lens of `v_l`; consistency
  across questions; steering with late vectors. Parked.
- **Is the direction necessary, not only sufficient?** → next experiment (ablation vs negative steering).
- Do the three traits share the MLP 18/19 peak because of a shared mechanism, or only because of the shared
  ruler effect (point 3)? Probably the latter; not tested.
- Layers 1–10 reconstruct with cos ≥ 0.96 instead of 0.999. Is that bf16 eager-vs-SDPA rounding? Not checked.
