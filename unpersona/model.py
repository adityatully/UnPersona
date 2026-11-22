"""Load the subject model (Qwen2.5-7B-Instruct) for interpretability work.  Env: `interp`.

Conventions used across this package (see CLAUDE.md):
  - HF / nnsight keep nn.Linear.weight as [out, in]  (output = W x, one token = column vector).
    Example: model.model.layers[19].self_attn.o_proj.weight  [3584, 3584] = [d_model, n_heads*d_head].
  - Activations come as [batch, pos, d_model] (tokens are rows here; transpose to match the notes).
  - hidden_states[l] = output of decoder block l-1 (l = 0 is the embedding output). See vectors.py.
"""
import torch
from nnsight import LanguageModel

MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Qwen2.5-7B-Instruct shape facts (from model.config):
N_LAYERS = 28          # decoder blocks; hidden_states has N_LAYERS + 1 = 29 entries
D_MODEL = 3584         # residual stream width
N_HEADS = 28           # query heads
N_KV_HEADS = 4         # key/value heads (grouped-query attention: 7 query heads share one KV head)
D_HEAD = 128           # N_HEADS * D_HEAD = 3584
D_MLP = 18944          # MLP hidden width (gate/up project to this, down projects back)


def load_model(name: str = MODEL, device_map: str = "auto") -> LanguageModel:
    """nnsight LanguageModel on the plain HF model: bf16 weights, eager attention.

    bf16: the weights are published in bf16, so fp32 adds no information to them; compute metrics
          in fp32 yourself (`.float()`) - projections, KL, score differences.
    eager attention: the fused SDPA/flash kernels never materialise the attention pattern
          [batch, n_heads, q_pos, k_pos]; eager does, so you can read and patch it.
    Weights ~15.2 GB (14.2 GiB measured); leaves ~8 GB of the A5000 for activations / KV cache.
    nnsight 0.8 warns that LanguageModel is deprecated in favour of TransformersModel; it still works,
    and the module tree (model.model.layers[i], model.lm_head) is the HF one either way.
    The plain HF module (no tracing) is model._module.
    """
    model = LanguageModel(
        name,
        dtype=torch.bfloat16,          # transformers >= 5 name for torch_dtype
        attn_implementation="eager",
        device_map=device_map,
        dispatch=True,                 # load the weights now, not at the first trace
    )
    model.eval()
    cfg = model.config
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads) == \
        (N_LAYERS, D_MODEL, N_HEADS, N_KV_HEADS), "unexpected model config"
    return model
