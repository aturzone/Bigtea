---
topic: The prompt to paste into a second worktree session, so two agents work in parallel without colliding
status: active
links: [lts-parity-criteria.md]
---

# How the split works

Two sessions, partitioned **by crate**, so neither touches the other's files and
the branches merge cleanly.

| | crates it owns | branch |
|---|---|---|
| **Main session** | `chaos-arch`, `chaos-model`, `chaos-io`, `chaos-plan`, `chaos-probe`, `chaos-ggml` | `ticket/r7-factored-experts` |
| **Worktree session** | `chaos-tokenizer`, `chaos-gguf` | `ticket/r8-tokenizers-and-containers` |

The one shared file is `docs/graph/backlog/lts-parity-criteria.md`. The worktree
session updates **only its own rows** (A6, D1, D2), which keeps the diff to
non-overlapping lines.

`STATUS.md` is owned by the main session — the worktree session records its
results in its own research node and its PR body instead, and the main session
folds them in.

## Setting it up

```bash
cd C:/Projects/Chaos
git worktree add ../Chaos-tok -b ticket/r8-tokenizers-and-containers
cd ../Chaos-tok
```

Then start Claude Code there and paste the prompt below.

---

# The prompt

Copy everything between the lines.

---

You are working on **Chaos**, a Rust CPU inference runner for GGUF models, in a
git worktree at `C:/Projects/Chaos-tok` on branch
`ticket/r8-tokenizers-and-containers`.

**Read `STATUS.md` and `docs/graph/backlog/lts-parity-criteria.md` first.** The
second file is the checklist that decides when v0.0.X LTS ships. The goal is
parity with llama.cpp on the models people actually run.

## Your scope — and its hard boundary

You own exactly two crates:

- **`crates/chaos-tokenizer`**
- **`crates/chaos-gguf`**

**Do not modify anything under `crates/chaos-arch`, `crates/chaos-model`,
`crates/chaos-io`, `crates/chaos-ggml`, `crates/chaos-plan` or
`crates/chaos-probe`.** Another session is working in those right now and a
change there will collide. If a task seems to need one, stop and write down what
you needed instead — do not reach across.

In `docs/graph/backlog/lts-parity-criteria.md` edit **only the A6, D1 and D2
rows**. Do not touch `STATUS.md` at all; put your results in a new node under
`docs/graph/research/` and in your PR body.

## Tickets, in order

**A6a — WPM tokenizer** (`tokenizer.ggml.model = "bert"`). WordPiece: lowercase
and strip accents when the container says so, split on punctuation and
whitespace, then greedy longest-match against the vocabulary with `##`
continuation pieces, and `[UNK]` when nothing matches. Unlocks the BERT family
and every embedding model.

**A6b — UGM tokenizer** (`tokenizer.ggml.model = "t5"`). SentencePiece Unigram:
a Viterbi lattice over the vocabulary maximising total score, with the
precompiled character map applied first. Unlocks T5 and mT5.

**D2 — GGUF v2 containers.** v3 works; v2 is untested. The difference is that v2
writes array lengths as `u32` where v3 uses `u64`. Find the version check in
`chaos-gguf`, handle both, and write a test that builds a v2 header in memory
and parses it — do **not** download a v2 model just for this.

**D1 — metadata robustness.** Build a small corpus of hand-written malformed
headers (truncated strings, an array length that overruns the file, an unknown
value type, a zero-length tensor name, duplicate keys) and assert that each is a
clean `Err`, never a panic and never a silent wrong value. This is a fuzz-style
test written by hand; no fuzzing crate — the workspace has **no external
dependencies** and that is deliberate.

**A6c — pre-tokenizer variants.** `tokenizer.ggml.pre` selects a splitting regex
and Chaos currently ignores it. `llama-bpe`, `deepseek-llm`, `qwen2` and
`falcon` differ in how they split digits and contractions, and the wrong one
shifts every token boundary. Read the value, implement the ones you can test
against a real container, and **refuse or warn loudly** on one you cannot.

## Rules

- **Commit and push after each ticket.** Do not batch them.
- Push with the token from `C:/Projects/.env`, inline in the URL, output
  redacted: `TOKEN=$(grep '^GITHUB_TOKEN=' /c/Projects/.env | cut -d= -f2-)` then
  `git push "https://${TOKEN}@github.com/aturzone/Chaos.git" <branch> 2>&1 | sed "s|${TOKEN}|[REDACTED]|g"`.
  **Never** echo the token, never put it in git config.
- `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all`
  must stay clean. All tests must pass.
- **A wrong tokenizer never crashes.** It produces different tokens, the model
  predicts a continuation of those tokens, and the output is fluent nonsense.
  So test the pieces separately — splitting, merging, byte fallback, round trip
  — and against a **real container** wherever one exists, not only against
  strings you invented for the matcher.
- Test against real vocabularies: `crates/chaos-tokenizer/tests/real_vocab.rs`
  already does this for BPE and SPM and is the pattern to follow. Mark
  container-backed tests `#[ignore]` with a reason.
- **Do not claim an architecture or tokenizer works until you have read its
  actual output.** Loading is not evidence. Gemma-2 loaded through this codebase
  with no error at all and answered "The capital of France is" with "himſelf".
- Internet is available. Download a small container when a ticket needs one to
  be verified rather than guessed — a BERT/embedding GGUF for WPM, a T5 GGUF for
  UGM. Put them in `C:/Projects/models/<name>/`.

## Build

```bash
export PATH="/c/msys64/mingw64/bin:$PATH"
export GGML_LIB_DIR=C:/Projects/llamacpp-unsloth/build/ggml/src
cargo test --release -p chaos-tokenizer -p chaos-gguf
```

Windows needs the **GNU** Rust toolchain. `.cargo/config.toml` already sets
`link-self-contained=no`; do not remove it.

## Working style

Do not stop after one tool call. Work through a ticket until it is committed and
pushed, then start the next. When you finish all five, open a PR against `main`
titled `tokenizers and container robustness` and summarise what was verified
against a real container versus only unit-tested.
