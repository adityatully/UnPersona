import torch
D = "/data/coe9/artifacts/unpersona/vectors/Qwen2.5-7B-Instruct/"
T = ["evil", "sycophantic", "hallucinating"]
V = {(t, v): torch.load(D + f"{t}_{v}_diff.pt") for t in T for v in ["response_avg", "prompt_avg", "prompt_last"]}
for (t, v), x in V.items():
    assert x.shape == (29, 3584) and torch.isfinite(x).all(), (t, v, x.shape)
print("all 9 files: shape", tuple(x.shape), "dtype", x.dtype, "finite")
print("\nresponse_avg norm per layer (row l = hidden_states[l]):")
print(" layer " + "".join(f"{t:>15s}" for t in T))
for l in [0, 1, 4, 8, 12, 16, 18, 20, 22, 24, 26, 28]:
    print(f"  {l:3d}  " + "".join(f"{V[(t,'response_avg')][l].norm():15.3f}" for t in T))
cos = torch.nn.functional.cosine_similarity
print("\ncosine between traits, response_avg:")
for l in [10, 20, 28]:
    a, b, c = (V[(t, "response_avg")][l].float() for t in T)
    print(f"  layer {l:2d}: evil-syco {cos(a,b,dim=0):.3f}  evil-hall {cos(a,c,dim=0):.3f}  syco-hall {cos(b,c,dim=0):.3f}")
print("\ncosine between variants at layer 20:")
for t in T:
    r, pa, pl = (V[(t, v)][20].float() for v in ["response_avg", "prompt_avg", "prompt_last"])
    print(f"  {t:13s} resp-prompt_avg {cos(r,pa,dim=0):.3f}  resp-prompt_last {cos(r,pl,dim=0):.3f}")
sub = torch.load("/data/coe9/artifacts/unpersona/vectors_check/bf16/evil_response_avg_diff.pt")
full = V[("evil", "response_avg")]
print("\nevil response_avg, 50-pair subset vs all 884 pairs: cos at layers 10/20/28 =",
      " ".join(f"{cos(sub[l], full[l], dim=0):.3f}" for l in [10, 20, 28]))
