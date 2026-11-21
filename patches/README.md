# patches/

Small changes to the paper's code (`third_party/persona_vectors`, upstream commit `b8e0f04`).
The upstream checkout is never edited. `make_run_copy.sh` copies it to
`/data/coe9/artifacts/unpersona/pv_run` and applies every `*.patch` here; the pipeline runs there.

| Patch | Files | What it changes |
|---|---|---|
| `0001-local-judge.patch` | `judge.py`, `config.py` | `JUDGE_SCORING=logprob` (default, paper's method) or `text` (parse the number; for judges like Qwen that split numbers into digits). In logprob mode, a startup check asks the local vLLM server how it tokenizes "85" and "100" and stops if either is more than one token. `HF_TOKEN` is no longer required (`hf auth login` is enough). |

Local judge: the OpenAI client reads `OPENAI_BASE_URL` (e.g. `http://localhost:8000/v1`) and
`OPENAI_API_KEY` (any dummy string; the local server does not check it). No code change needed for that.
