# Contributing to Bigtea

Thanks for looking. Bigtea is a small project with an unusual amount of written
reasoning behind it, and the fastest way to be useful is to read a little of it
before writing code.

## The one rule that matters most

> **A performance claim is not citable until the competitor's exact command line
> and its output are in a document.**

This exists because the project broke it. "llama.cpp cannot run a model larger
than RAM" survived for days on a misattributed error string, because nobody ran
the opposing command. When someone finally did, llama.cpp ran the 144 GB model
fine with `--no-repack`, and the project's central marketing claim evaporated.

So: run it, paste it, and flag it. A benchmark without the losing command line is
an opinion.

The same applies to your own changes. "This should be faster" is not a result.
The project's record on reasoning ahead of measurement is **nought for four**:

| predicted | measured |
|---|---|
| Expert reads are latency-bound; parallelise them | 14% *slower*, reverted |
| Contextual sparsity is worth 5-10x | V4-Flash experts are 9.1% sparse, not 80% |
| Residency is "likely a large multiple" | 22% |
| Threadpool churn explains the flat thread curve | 1 thread is 4.7x *slower*, not faster |

## Before you start

1. Read [`docs/graph/INDEX.md`](docs/graph/INDEX.md), then **only the two or three
   nodes your task links to**. The graph is large; reading it all is a waste.
2. Read [`CLAUDE.md`](CLAUDE.md). It is the project's short list of facts that
   cost real time to rediscover — things like *ggml aborts the process when its
   arena is exhausted*, and *a wrong tokenizer produces fluent nonsense, never a
   crash*.

## Building and testing

```bash
export GGML_LIB_DIR=/path/to/llama.cpp/build/ggml/src
cargo build --release
cargo test --release              # 157 tests
cargo test --release -- --ignored # 16 more, need a real model on disk
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The `--ignored` tests read from real GGUF containers and are skipped silently
when the file is absent, so they are safe to run without the model. If you have
one, run them: they are what verify the forward pass against llama.cpp's own
element sums, and they catch the class of bug that produces plausible output.

## What good looks like here

**Correctness is checked, not assumed.** In this domain a wrong forward pass, a
wrong tokenizer, a wrong mask or a wrong cache all produce *fluent, plausible
text* rather than an error. Every numerical change needs a test that would fail
if the numbers were wrong — ideally against an external oracle, which for us is
`llama-eval-callback`'s element sums.

**Comments explain why, not what.** The codebase is written for someone reading
it in a year with no memory of the session that produced it. If something looks
strange, the comment should say what it prevents. If a value was chosen by
measurement, say what was measured.

**Negative results are kept.** If you try something and it does not work, a
commit or a doc node recording *what you measured and why it failed* is more
valuable than silence. Several of this project's best decisions came from
written-down failures.

## Pull requests

- Branch from `main` as `ticket/<short-name>`.
- One concern per PR.
- Say in the description what you measured, on what hardware, and how.
- CI must be green: build and unit tests on Linux, macOS and Windows.

Note that **Linux and macOS are currently untested at runtime.** The code has
per-platform direct-I/O paths (`O_DIRECT`, `F_NOCACHE`, `FILE_FLAG_NO_BUFFERING`)
and only the Windows one has ever been executed. If you run Bigtea on Linux or
macOS — even if it just fails — that is a genuinely useful bug report.

## Good first issues

- Run it on Linux or macOS and report what happens.
- `bigtea-probe` on unusual hardware: report what it gets wrong.
- Documentation: if something in the README did not work, that is a bug.
- Architectures: adding one is well-bounded work with a clear oracle to check
  against. `crates/bigtea-arch/src/qwen3.rs` is the smaller example to copy.

## Reporting a bug

Include: OS, CPU, RAM, the model and quant, the exact command, and the full
output. If it is a *wrong output* bug rather than a crash, say what you expected
and what you got — those are the hardest and most valuable ones.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

Contributions are accepted under the Apache License 2.0, matching the project.
