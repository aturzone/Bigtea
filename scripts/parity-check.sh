#!/usr/bin/env bash
# Diff Bigtea's output against llama.cpp's, token for token, at --temp 0.
#
# This exists because `VERIFIED_ARCHITECTURES` once contained an entry nobody
# had ever run the reference against: gemma2 answered "**Paris**." where
# llama.cpp answered ":\n\na) Paris\nb) Lyon", for weeks, because "it loads and
# it answers in English" was mistaken for a check. Loading is not evidence.
#
#   scripts/parity-check.sh <model.gguf> [n_tokens]
#
# Exits non-zero if any prompt disagrees *and* the reference agrees with
# itself. Needs LLAMACPP_BIN pointing at a llama.cpp build's bin/.
#
# # Why a mismatch is not automatically a bug
#
# Greedy decoding is not stable under changes that are mathematically no-ops.
# llama.cpp itself answers `def fibonacci(n):` two different ways on
# Llama-3.2-1B depending on `-fa on|off`, and two different ways on Phi-3
# depending on `--no-repack`. Those prompts sit on a near-tie: any
# implementation that reorders a sum lands on the other side and then writes a
# different paragraph. Calling that a failure would have sent someone hunting a
# bug that is not there -- so every mismatch is re-run against a second
# reference configuration first, and only a mismatch the reference is stable
# across counts.
set -u

MODEL=${1:?usage: parity-check.sh <model.gguf> [n_tokens]}
N=${2:-32}
LLAMACPP_BIN=${LLAMACPP_BIN:-/c/Projects/llamacpp-unsloth/build/bin}
BIGTEA=${BIGTEA:-./target/release/bigtea-run.exe}
REF="$LLAMACPP_BIN/llama-completion.exe"

# Deliberately three kinds of text. A single factual prompt hides an activation
# error: "The capital of France is Paris" survives a surprising amount of wrong
# arithmetic, because the answer is overdetermined by the training data.
PROMPTS=(
  "The capital of France is"
  "Once upon a time"
  "def fibonacci(n):"
)

strip() { sed 's/\x1b\[[0-9;]*m//g' | tr -d '\r'; }

ref() { "$REF" -m "$MODEL" -p "$1" -n "$N" --temp 0 --no-warmup -no-cnv "${@:2}" 2>/dev/null | strip; }

name=$(basename "$MODEL")
fail=0
unstable=0
for p in "${PROMPTS[@]}"; do
  a=$("$BIGTEA" -m "$MODEL" -p "$p" -n "$N" --temp 0 --force 2>/dev/null | strip)
  b=$(ref "$p")
  a=${a#*"$p"}
  b=${b#*"$p"}
  # Bigtea puts the echo on its own line; llama.cpp echoes inline. So exactly
  # one newline on Bigtea's side is framing rather than output -- and stripping
  # it from BOTH is wrong, because a model whose first token is a newline (the
  # `def fibonacci(n):` prompt, on every model tested) then loses a real token
  # from llama.cpp's side and every engine looks like a failure.
  a=${a#$'\n'}

  if [ "$a" = "$b" ]; then
    printf 'ok        %-36s %s\n' "$name" "$p"
    continue
  fi

  # Ask the reference whether it agrees with itself. `-fa off` and
  # `--no-repack` both change only how a sum is ordered.
  c=$(ref "$p" -fa off)
  d=$(ref "$p" --no-repack)
  c=${c#*"$p"}
  d=${d#*"$p"}
  if [ "$c" != "$b" ] || [ "$d" != "$b" ]; then
    unstable=$((unstable + 1))
    printf 'unstable  %-36s %s\n' "$name" "$p"
    printf '  the reference disagrees with itself here (-fa off / --no-repack),\n'
    printf '  so this prompt is a near-tie and proves nothing either way.\n'
    continue
  fi

  fail=1
  printf 'FAIL      %-36s %s\n' "$name" "$p"
  printf '  bigtea   : %s\n' "$(printf '%s' "$a" | head -c 200)"
  printf '  llama.cpp: %s\n' "$(printf '%s' "$b" | head -c 200)"
done

[ "$unstable" -gt 0 ] && printf '          %-36s %d prompt(s) unstable in the reference itself\n' "$name" "$unstable"
exit $fail
