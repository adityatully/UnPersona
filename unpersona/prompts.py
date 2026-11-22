"""Chat prompts, questions and the graded system prompts.

Trap (Qwen chat template): if you pass NO system message, the template inserts Qwen's default one,
"You are Qwen, created by Alibaba Cloud. You are a helpful assistant." So "no system prompt" is a
specific system prompt. build_prompt(None, ...) keeps that default (it is what the paper's unsteered
eval runs saw); pass system="" if you really want an empty system turn.
"""
import json
import os

PV_DATA = os.environ.get("PV_DATA", "/data/coe9/unpersona/third_party/persona_vectors/data_generation")
GRADED = os.environ.get("UNPERSONA_GRADED", "/data/coe9/unpersona/data/graded_prompts.json")


def build_prompt(tokenizer, user: str, system: str | None = None) -> str:
    """Chat-templated prompt string, ending where the assistant starts writing.

    add_generation_prompt=True appends '<|im_start|>assistant\\n', so the model's next token is the
    first answer token. Tokenize the result with add_special_tokens=False: the template already has
    every special token, and Qwen has no BOS token.
    """
    messages = ([{"role": "system", "content": system}] if system is not None else []) + \
               [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def last_prompt_pos(tokenizer, prompt: str) -> int:
    """Index of the last prompt token in tokenizer(prompt, add_special_tokens=False).input_ids.

    For a build_prompt() string this token is '\\n' after 'assistant' (the token the paper's
    prompt_last vectors and section 3.3 projections use). With a batch of LEFT-padded prompts the
    last prompt token is simply position -1; with right padding it is this index per prompt.
    """
    ids = tokenizer(prompt, add_special_tokens=False).input_ids  # [n_tokens]
    return len(ids) - 1


def load_questions(trait: str, split: str = "eval") -> list[str]:
    """The paper's 20 questions for a trait. split='extract' built the vectors; 'eval' is held out."""
    assert split in ("extract", "eval"), split
    with open(os.path.join(PV_DATA, f"trait_data_{split}", f"{trait}.json")) as f:
        return json.load(f)["questions"]


def load_instructions(trait: str) -> list[dict]:
    """The 5 extraction system-prompt pairs [{'pos': ..., 'neg': ...}]. The generation code prefixes
    'You are a(n) <name> assistant. ' (name = trait for pos; we used 'helpful' for every neg)."""
    with open(os.path.join(PV_DATA, "trait_data_extract", f"{trait}.json")) as f:
        return json.load(f)["instruction"]


def load_graded_prompts(trait: str) -> list[str]:
    """8 system prompts from paper appendix C.3; index = trait encouragement (0 least, 7 most).
    See data/README.md (the paper numbers them the other way round)."""
    with open(GRADED) as f:
        return json.load(f)[trait]
