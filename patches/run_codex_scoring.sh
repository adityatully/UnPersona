#!/usr/bin/env bash
# Score the rescoring documents with the Codex CLI (ChatGPT login, no API key), one fresh session per doc.
# Usage: bash run_codex_scoring.sh [FIRST] [LAST]      e.g. 3 6   (doc numbers; default 1 84)
#   MODEL=<name> to pick a model (default: the one in ~/.codex/config.toml)
# Skips docs whose returned/<doc>.csv already exists. Logs each session's events to logs/<doc>.jsonl
# so we can verify the model ran no commands.
set -uo pipefail
ROOT=${ROOT:-$HOME/Downloads/unpersona_rescore}
FIRST=${1:-1}; LAST=${2:-84}
EMPTY=$(mktemp -d)          # empty working dir: no repo, no AGENTS.md, nothing to read
mkdir -p "$ROOT/returned" "$ROOT/logs"
PROMPT='The <stdin> block is a complete scoring task. Do it exactly as written. Do not run any commands, do not read or write any files, do not use tools: read every item yourself. Your final message must be only the CSV block the task asks for.'
for f in "$ROOT"/docs/*.md; do
  doc=$(basename "$f" .md); n=$((10#${doc%%_*}))
  [ "$n" -lt "$FIRST" ] || [ "$n" -gt "$LAST" ] && continue
  out="$ROOT/returned/$doc.csv"
  if [ -s "$out" ]; then echo "skip $doc"; continue; fi
  echo "$(date '+%T') $doc"
  codex exec --ephemeral --skip-git-repo-check -s read-only -C "$EMPTY" ${MODEL:+-m "$MODEL"} \
      --json -o "$out.tmp" "$PROMPT" < "$f" > "$ROOT/logs/$doc.jsonl" 2> "$ROOT/logs/$doc.err"
  if [ -s "$out.tmp" ] && grep -q "," "$out.tmp"; then
    mv "$out.tmp" "$out"; echo "   saved $(grep -cE '^[a-z0-9]{5},' "$out") lines"
  else
    echo "   FAILED (see logs/$doc.err)"; rm -f "$out.tmp"
  fi
done
rmdir "$EMPTY" 2>/dev/null
