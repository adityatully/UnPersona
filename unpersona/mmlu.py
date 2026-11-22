"""A fixed MMLU subset and a plain (no intervention) accuracy function.

Scoring: zero-shot, chat template, the model's next-token logits at the last prompt position,
restricted to the four tokens 'A' 'B' 'C' 'D'; prediction = argmax over those four. No generation,
so no sampling noise and no answer parsing. Qwen2.5-7B-Instruct's reported MMLU is ~74% (5-shot,
base-model style); expect the same ballpark here, not the same number.
"""
import numpy as np
import torch

from .prompts import build_prompt

LETTERS = ["A", "B", "C", "D"]


def load_mmlu_subset(n: int = 500, seed: int = 0) -> list[dict]:
    """n questions sampled (without replacement, fixed seed) from the MMLU test split, all 57 subjects.
    Each item: {'question', 'choices' [4], 'answer' (0-3), 'subject'}."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")  # 14,042 questions
    idx = np.random.default_rng(seed).choice(len(ds), size=n, replace=False)
    return [dict(question=ds[int(i)]["question"], choices=ds[int(i)]["choices"],
                 answer=int(ds[int(i)]["answer"]), subject=ds[int(i)]["subject"]) for i in idx]


def mmlu_user_message(item: dict) -> str:
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, item["choices"]))
    return f"{item['question']}\n\n{opts}\n\nAnswer with the letter of the correct option only."


@torch.no_grad()
def mmlu_accuracy(model, items: list[dict], system: str | None = None, batch_size: int = 8) -> dict:
    """Plain accuracy of an nnsight LanguageModel (no hooks, no steering).

    Per batch:
      prompts            B strings (build_prompt), left-padded so the last prompt token is at pos -1
      logits             [B, pos, vocab=152064]  from model.lm_head.output
      logits[:, -1, ids] [B, 4]  scores for 'A' 'B' 'C' 'D'  (computed in fp32)
    Returns {'accuracy', 'n', 'pred' [n], 'p_correct' [n]} (p_correct = softmax over the 4 letters).
    """
    tok = model.tokenizer
    tok.padding_side = "left"
    letter_ids = [tok(l, add_special_tokens=False).input_ids for l in LETTERS]
    assert all(len(x) == 1 for x in letter_ids), letter_ids  # each letter is one token
    letter_ids = [x[0] for x in letter_ids]
    preds, p_correct = [], []
    for s in range(0, len(items), batch_size):
        batch = items[s:s + batch_size]
        prompts = [build_prompt(tok, mmlu_user_message(it), system) for it in batch]
        with model.trace(prompts, add_special_tokens=False):
            logits = model.lm_head.output.save()          # [B, pos, vocab]
        four = logits[:, -1, letter_ids].float()          # [B, 4]
        probs = four.softmax(-1)                          # [B, 4]
        preds += four.argmax(-1).tolist()
        p_correct += [probs[i, it["answer"]].item() for i, it in enumerate(batch)]
    gold = [it["answer"] for it in items]
    return dict(accuracy=float(np.mean(np.array(preds) == np.array(gold))), n=len(items),
                pred=preds, p_correct=p_correct)
