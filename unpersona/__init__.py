"""unpersona helpers (boilerplate only; experiments live in experiments/).

Import the submodule you need; nothing is imported here, so the judge tools still run in the
`persona` env, which has no nnsight.

    model.py        load_model()                     env interp
    vectors.py      load_vectors(trait, variant)     layer index convention in its docstring
    prompts.py      build_prompt, last_prompt_pos, load_questions, load_instructions, load_graded_prompts
    mmlu.py         load_mmlu_subset, mmlu_accuracy  env interp
    judge.py        judge_file (local vLLM judge)    env persona
    calibration.py, rescore.py                       judge validation and second-judge rescoring
"""
