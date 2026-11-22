# GPU test of the task-6 helpers: load_model, nnsight hidden states == HF hidden_states, MMLU subset.
import time, torch
from unpersona.model import load_model
from unpersona.prompts import build_prompt, last_prompt_pos
from unpersona.mmlu import load_mmlu_subset, mmlu_accuracy
t0 = time.time()
model = load_model()
print(f"loaded in {time.time()-t0:.0f}s; dtype {next(model.parameters()).dtype}; "
      f"attn {model.config._attn_implementation}; GPU mem {torch.cuda.memory_allocated()/2**30:.1f} GiB")
tok = model.tokenizer
p = build_prompt(tok, "What should be done with criminals in society?")
# nnsight: output of block 19 (= hidden_states[20]) vs HF output_hidden_states on the same input
with model.trace(p, add_special_tokens=False):
    h20 = model.model.layers[19].output.save()               # nnsight 0.8: block output, [1, pos, 3584]
ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    hs = model._module(ids, output_hidden_states=True).hidden_states   # tuple of 29 x [1, pos, 3584]
h20 = h20[0] if isinstance(h20, tuple) else h20
print("block-19 output shape", tuple(h20.shape), " max |nnsight - HF hidden_states[20]| =",
      (h20.float() - hs[20].float()).abs().max().item())
print("last prompt pos", last_prompt_pos(tok, p), "of", ids.shape[1])
items = load_mmlu_subset(500, seed=0)
t0 = time.time()
r = mmlu_accuracy(model, items, batch_size=8)
print(f"MMLU subset n={r['n']}  accuracy {r['accuracy']:.3f}  mean p(correct) {sum(r['p_correct'])/r['n']:.3f}  ({time.time()-t0:.0f}s)")
