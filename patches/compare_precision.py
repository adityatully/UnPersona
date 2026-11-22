# Compare evil vectors from the same 50 pairs: bf16 forward (fp32 averaging) vs fp32 forward.
import torch
D = "/data/coe9/artifacts/unpersona/vectors_check/"
for var in ["response_avg", "prompt_avg", "prompt_last"]:
    a = torch.load(D + f"bf16/evil_{var}_diff.pt").float()   # [29, 3584]
    b = torch.load(D + f"fp32/evil_{var}_diff.pt").float()   # [29, 3584]
    assert a.shape == b.shape == (29, 3584), (a.shape, b.shape)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)  # [29]
    ratio = a.norm(dim=1) / b.norm(dim=1)                     # [29]
    rel = (a - b).norm(dim=1) / b.norm(dim=1)                 # [29] relative error of the whole vector
    print(f"\n{var}:  min cos (layers 1-28) {cos[1:].min():.5f}   max |norm ratio - 1| {(ratio[1:]-1).abs().max():.4f}")
    print(" layer   cos       norm_bf16  norm_fp32  rel_err")
    for l in [0, 1, 5, 10, 15, 20, 25, 28]:
        print(f"  {l:3d}  {cos[l]:.6f}  {a[l].norm():9.3f}  {b[l].norm():9.3f}  {rel[l]:.4f}")
