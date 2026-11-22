"""Load the persona vectors built in task 3.

Layer index convention (checked against generate_vec.py, task 3):
    vec[l]  is the mean difference of hidden_states[l]  (HF output_hidden_states=True)
    hidden_states[0]  = embedding output (before block 0)
    hidden_states[l]  = output of decoder block l-1  = input of block l     (l = 1..28)
    "layer 20" in the paper = vec[20] = output of block 19 = model.model.layers[19] output;
    the paper's ActivationSteerer hooks layers[layer - 1] for this reason.
    vec[28] = output of the last block, BEFORE the final RMSNorm (model.model.norm).

Vectors are NOT normalised: ||vec[l]|| grows with depth like the residual stream (evil response_avg:
~0.05 at l=0, ~27 at l=20, ~96 at l=28). Normalise yourself when a formula needs a unit vector.
"""
import os

import torch

VEC_DIR = os.environ.get("UNPERSONA_VECTORS", "/data/coe9/artifacts/unpersona/vectors/Qwen2.5-7B-Instruct")
TRAITS = ("evil", "sycophantic", "hallucinating")
VARIANTS = ("response_avg", "prompt_avg", "prompt_last")


def load_vectors(trait: str, variant: str = "response_avg") -> torch.Tensor:
    """Return the persona vector for all layers, fp32 on CPU.

    returns  vec  [n_layers + 1, d_model] = [29, 3584]   row l lives in the space of hidden_states[l]

    variant (how each answer was summarised before the pos - neg mean):
      response_avg  mean over the answer tokens          (the paper's main vector, used for steering)
      prompt_avg    mean over the prompt tokens          (includes the system prompt's own words;
                    its length is precision-sensitive in layers 5-15, use its direction)
      prompt_last   the last prompt token only           ('\\n' after '<|im_start|>assistant');
                    row 0 is exactly 0 (same token, same embedding in pos and neg)
    """
    assert trait in TRAITS and variant in VARIANTS, (trait, variant)
    vec = torch.load(os.path.join(VEC_DIR, f"{trait}_{variant}_diff.pt"), map_location="cpu").float()
    assert vec.shape == (29, 3584), vec.shape
    return vec  # [29, 3584]
