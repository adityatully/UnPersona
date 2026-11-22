# Does tokenizing prompt+answer start with exactly the tokens of the prompt alone?
# generate_vec takes prompt_len from the prompt alone; a merge at the boundary would shift the split.
import pandas as pd
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
D = "/data/coe9/artifacts/unpersona/judged_final/Qwen2.5-7B-Instruct/"
for t in ["evil", "sycophantic", "hallucinating"]:
    p, n = pd.read_csv(D + f"{t}_pos_instruct.csv"), pd.read_csv(D + f"{t}_neg_instruct.csv")
    m = (p[t] >= 50) & (n[t] < 50) & (p.coherence >= 50) & (n.coherence >= 50)
    bad, total = 0, 0
    for d in (p[m], n[m]):
        for pr, a in zip(d.prompt, d.answer.fillna("")):
            full = tok(pr + a, add_special_tokens=False).input_ids
            pi = tok(pr, add_special_tokens=False).input_ids
            total += 1
            bad += full[:len(pi)] != pi
    print(f"{t:13s} answers {total}  boundary mismatches {bad}")
ex = p.prompt.iloc[0]
print("prompt ends with:", repr(ex[-60:]))
print("first token has BOS?", tok(ex, add_special_tokens=False).input_ids[:3], tok.bos_token)
