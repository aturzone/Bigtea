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
# # `unstable` is a SUSPICION, not a verdict
#
# The re-check below compares the reference TO ITSELF under flags that only
# reorder a sum. It cannot see that OUR INPUT differed -- and when the input
# differs, a near-tie is exactly the symptom, because the model is answering a
# slightly different question and lands on the other side of whatever was close.
#
# **Nine of eleven `unstable` verdicts in one session turned out to be bugs**:
# Llama-3.2 rotating with the wrong RoPE (4), Falcon3 prefilled a token short
# (5). So this script now does two more things before shrugging:
#
#   1. **Compares the tokenized prompt, ID for ID.** If llama.cpp turns the
#      prompt into different tokens than Bigtea does, the two engines are not
#      answering the same question and the mismatch is a FAILURE, not a tie.
#      **This compared COUNTS until 2026-08-15, and a count cannot see the bug
#      it was written for**: `starcoder2` passed 3/3 while running the wrong
#      pre-tokenizer, and a different split of the same text usually produces
#      the same *number* of tokens. A missing BOS moves the count and was
#      caught; a wrong merge table does not and was not.
#   2. **Counts them.** One near-tie in eight is ordinary. Three or more is a
#      bug that has not been found yet, and the script exits non-zero saying so
#      rather than printing eight reassuring lines.
#   3. **Asks whether OUR answer is inside the band.** "The reference disagrees
#      with itself" and "our output is one of the things it disagrees between"
#      are different questions, and for a day this script asked the first while
#      reporting it as though it answered the second. Reproducing llama.cpp's
#      own `-b 1` output byte for byte is near-conclusive; producing a third
#      answer it never gives under any no-op is not explained by its
#      instability at all. Those are now `near-tie` and `unstable`.
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

# Eight prompts, deliberately unalike, and the count is the point.
#
# A PASS IS EVIDENCE ABOUT THESE PROMPTS, NOT ABOUT THE ARCHITECTURE. That is
# not pedantry: `starcoder2` passed 3/3 while running the WRONG PRE-TOKENIZER,
# and only agreed because its merge table happened to differ from the model
# that failed. Three prompts were enough to certify an architecture and not
# enough to notice that its input was being split wrongly.
#
# A single factual prompt is the weakest of all: "The capital of France is
# Paris" survives a surprising amount of wrong arithmetic, because the answer
# is overdetermined by the training data. It was the code prompt that exposed
# the pre-tokenizer, and the code prompt again that exposed Gemma's activation.
# So: prose, code, a numeric run, a list continuation, arithmetic, SQL, and
# formal register -- each stresses a different part of the vocabulary and a
# different part of the graph.
PROMPTS=(
  "The capital of France is"
  "Once upon a time"
  "def fibonacci(n):"
  "1 2 3 4 5 6 7 8 9 10 11"
  "The following is a list of items: apples, oranges,"
  "Q: What is 17 plus 25? A:"
  "SELECT name, COUNT(*) FROM users WHERE"
  "Dear Sir or Madam, I am writing to"
)

# Strip terminal colouring, CRs, and llama.cpp's own end-of-stream marker.
#
# `[end of text]` is printed by llama-completion when it hits EOS; Bigtea stops
# silently. The GENERATED TOKENS ARE IDENTICAL -- this is the two CLIs framing
# the same result differently, and leaving it in reported tinyllama's
# `Q: What is 17 plus 25? A:` as a FAIL where both had answered ` 42`.
#
# A harness that cries wolf is worse than no harness. The whole value of this
# script is that a FAIL means something, and the first thing anyone does with a
# FAIL is go looking in the forward pass.
strip() {
  sed 's/\x1b\[[0-9;]*m//g' \
    | tr -d '\r' \
    | sed 's/ *\[end of text\] *$//'
}

ref() {
  "$REF" -m "$MODEL" -p "$1" -n "$N" --temp 0 --no-warmup -no-cnv "${@:2}" 2>/dev/null | strip
}

# The token IDs each engine makes of a prompt, as `1,450,7483,...`.
#
# **The IDs, not the count.** This compared counts, and a count cannot see the
# failure it was written for. `starcoder2` passed 3/3 while running the WRONG
# PRE-TOKENIZER — a different split of the same text usually yields the same
# number of tokens, so the check was blind to exactly the bug that motivated it.
# A missing BOS changes the count and was caught; a wrong merge table does not.
#
# Both engines print the IDs under `--verbose-prompt`; llama.cpp one per line as
# `<id> -> '<text>'`, prefixed with its log timestamp. Matching on the arrow and
# taking the number before it survives token texts containing quotes.
bigtea_tokens() {
  "$BIGTEA" -m "$MODEL" -p "$1" -n 1 --temp 0 --force --verbose-prompt 2>&1 \
    | strip | sed -n 's/^prompt  *[0-9]* tokens: \[\(.*\)\]$/\1/p' | head -1 | tr -d ' '
}
llama_tokens() {
  "$REF" -m "$MODEL" -p "$1" -n 1 --temp 0 --no-warmup -no-cnv --verbose-prompt 2>&1 \
    | strip | grep -oE "^[0-9.]+ I +[0-9]+ -> " | grep -oE '[0-9]+ -> $' \
    | grep -oE '^[0-9]+' | paste -sd, -
}

name=$(basename "$MODEL")
fail=0
unstable=0
near=0
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
  # `-b 1` is the third no-op, and adding it was a deliberate decision rather
  # than a convenience. Batching changes how many tokens a forward pass covers;
  # for a correct engine that only reorders sums. llama.cpp disagrees with
  # ITSELF under it -- on Qwen3-30B-A3B, `The capital of France is`:
  #
  #   default : ...Spain is Madrid. The capital of Germany is Berlin.
  #   -b 1    : ...Spain is Madrid. The capital of Portugal is Lisbon.
  #
  # THE SET OF NO-OP CONFIGURATIONS TESTED HERE DECIDES WHAT COUNTS AS A BUG,
  # and it cuts both ways: every configuration added makes `unstable` easier to
  # reach, and `unstable` is where a real bug hides. Llama-3.2 reported FOUR
  # unstable prompts for a day and all four were `rope_freqs.weight` being
  # ignored -- the cluster was the signal, not the noise.
  #
  # So this stays honest only because of the cluster rule below: three or more
  # unstable in eight still exits non-zero and demands a look.
  e=$(ref "$p" -b 1)
  # A COMPOSITION, because probing flags singly cannot see a prompt that only
  # moves when two are combined. Measured on Phi-3, `The capital of France is`:
  #
  #   -b 1         : Paris. Paris is known for its rich history,
  #   -fa off      : Paris.<|assistant|> Yes, that's correct
  #   -b 1 -fa off : Paris.<|assistant|> That's correct! Paris   <- Bigtea's answer
  #
  # Neither flag alone reproduces it. A near-tie that needs two no-ops composed
  # is invisible to a single-flag probe, and there is no reason to think that
  # class is rare -- it is just the class nobody looked for.
  f=$(ref "$p" -b 1 -fa off)
  c=${c#*"$p"}
  d=${d#*"$p"}
  e=${e#*"$p"}
  f=${f#*"$p"}
  if [ "$c" != "$b" ] || [ "$d" != "$b" ] || [ "$e" != "$b" ] || [ "$f" != "$b" ]; then
    # Before shrugging: did the two engines even read the same prompt? A
    # near-tie is what a DIFFERENT INPUT looks like, so this is checked first.
    bt=$(bigtea_tokens "$p")
    lt=$(llama_tokens "$p")
    if [ -n "$bt" ] && [ -n "$lt" ] && [ "$bt" != "$lt" ]; then
      fail=1
      printf 'FAIL      %-36s %s\n' "$name" "$p"
      printf '  the prompt tokenized differently -- the IDs, not just the count:\n'
      printf '    bigtea   : %s\n' "$(printf '%s' "$bt" | head -c 160)"
      printf '    llama.cpp: %s\n' "$(printf '%s' "$lt" | head -c 160)"
      printf '  The reference also disagrees with itself here, which is what a\n'
      printf '  different input looks like -- so this is NOT a near-tie.\n'
      continue
    fi
    which=""
    [ "$c" != "$b" ] && which="$which -fa-off"
    [ "$d" != "$b" ] && which="$which --no-repack"
    [ "$e" != "$b" ] && which="$which -b-1"
    [ "$f" != "$b" ] && which="$which -b-1+-fa-off"
    # WHICH configuration moved it, not merely that one did. "-b 1 only" is a
    # weaker claim than "every no-op moves it", and collapsing the two into one
    # word is how a cluster stops looking like a cluster.

    # **Is OUR answer inside the band the reference spans?** This is a different
    # question from "does the reference disagree with itself", and only the
    # second one discriminates. The old code asked the first and reported it as
    # if it answered the second: knowing llama.cpp is unstable here says nothing
    # about whether OUR output is one of the things it is unstable BETWEEN.
    #
    # Landing on a value llama.cpp itself produces under a no-op is the
    # strongest evidence available short of identity -- the answer is not merely
    # plausible, it is the reference's own. Landing on a value it never produces
    # under any of them is a different situation wearing the same word, and its
    # instability no longer explains our disagreement at all.
    inband=""
    [ "$a" = "$c" ] && inband="-fa off"
    [ "$a" = "$d" ] && inband="--no-repack"
    [ "$a" = "$e" ] && inband="-b 1"
    [ "$a" = "$f" ] && inband="-b 1 -fa off"
    if [ -n "$inband" ]; then
      near=$((near + 1))
      printf 'near-tie  %-36s %s\n' "$name" "$p"
      printf '  the reference disagrees with itself under:%s\n' "$which"
      printf '  and Bigtea reproduces its `%s` output EXACTLY. Our answer is\n' "$inband"
      printf '  one llama.cpp itself gives; this is a tie, not a divergence.\n'
      continue
    fi
    unstable=$((unstable + 1))
    printf 'unstable  %-36s %s\n' "$name" "$p"
    printf '  the reference disagrees with itself under:%s\n' "$which"
    printf '  but Bigtea matches NONE of its outputs -- a third answer, outside\n'
    printf '  the band. Its instability does not explain ours. Both engines\n'
    printf '  tokenized the prompt identically. Suspicious.\n'
    continue
  fi

  fail=1
  printf 'FAIL      %-36s %s\n' "$name" "$p"
  printf '  bigtea   : %s\n' "$(printf '%s' "$a" | head -c 200)"
  printf '  llama.cpp: %s\n' "$(printf '%s' "$b" | head -c 200)"
done

if [ "$near" -gt 0 ] || [ "$unstable" -gt 0 ]; then
  printf '          %-36s %d of %d in-band, %d outside\n' \
    "$name" "$near" "${#PROMPTS[@]}" "$unstable"
fi
# The cluster rule now applies to the SHARPER class. Nine of eleven `unstable`
# verdicts in one session were real bugs -- but that was measured when a single
# word covered both situations. A prompt where Bigtea reproduces one of the
# reference's own outputs byte for byte is explained; a prompt where it produces
# a third answer nobody's reference gives is not, and only the second kind is
# the thing those nine turned out to be.
if [ "$unstable" -ge 3 ]; then
  printf '          %-36s %d outside-band answers is a cluster, not chance -- treat as a bug\n' \
    "$name" "$unstable"
  fail=1
fi
# **In-band is not a blank cheque.** Every configuration added widens the band,
# so "in band" gets easier to reach as the probe grows -- exactly the direction
# that turns a harness into a rubber stamp. A model that ties on almost every
# prompt is systematically different even when each individual tie is explained,
# so the count is bounded too.
if [ "$near" -ge 6 ]; then
  printf '          %-36s %d of %d prompts tie -- in-band each time, but that is\n' \
    "$name" "$near" "${#PROMPTS[@]}"
  printf '          %-36s not what a matching engine looks like\n' ""
  fail=1
fi
exit $fail
