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

## What I learned
_(to be written by me)_

Draft points from the discussion:
- DLA onto a fixed readout direction is exact here and needs no normalization for L < 28; at L = 28 HF's
  hidden state is already normed, so the final-norm scale has to be frozen (level 1).
- Extract and eval give the same picture because they share the same 5 system prompt pairs — eval is held
  out on the questions, not on the system prompts that drive prompt_last.
- Direct writes onto v̂_L come mostly from the last 1–2 blocks before L. Because the direction rotates with
  depth, DLA onto one readout favours late components by construction: early blocks write earlier versions of
  the direction that are mostly orthogonal to v̂_20. "Block 19 creates the evil direction" does not follow
  (level 3); "blocks 18–19 write most of the component of the difference that lies along v̂_20" does (level 1).
- At this position attention is the only entry point for the system prompt; MLP contributions transform what
  attention imported earlier (level 1), so the MLP share is a direct-path share, not the origin.

## Open questions
_(to be written by me)_
