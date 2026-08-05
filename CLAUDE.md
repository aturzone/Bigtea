# Bigtea — a runner for models larger than RAM

- **What it is**: a Rust inference runner whose job is running models that do *not* fit in memory. Keeps the always-read weights resident, streams routed experts from disk per token. Borrows `ggml` for arithmetic; owns memory, residency, streaming, and the token loop.
- **Proven**: Qwen3-30B-A3B (17.28 GiB container) generates correct text on a 15.7 GiB machine holding 0.93 GiB resident + a 6.26 GiB expert cache.
- **We are still behind llama.cpp — do not claim otherwise.** On the same box it is ~1.2–2.4x faster at prefill and ~2.2–2.5x at generation, and it runs the 144 GB V4-Flash too once `--no-repack` is passed. "Larger than RAM" is not a differentiator; mmap already does it. Full ladder, retracted claims, and one experiment that failed: `docs/graph/research/head-to-head-llamacpp-2026-08-05.md`.
- Graph docs live in `/docs/graph/`; read `INDEX.md` first, then only the 2–3 nodes a task links to.

## Build / test / run

```
# ggml must be built first; point GGML_LIB_DIR at ggml-base.a, ggml-cpu.a, ggml.a
export GGML_LIB_DIR=C:/Projects/llamacpp-unsloth/build/ggml/src   # PowerShell: $env:GGML_LIB_DIR=...
cargo test --release          # 138 tests
cargo build --release
./target/release/bigtea-run <model.gguf> "prompt" -n 16
./target/release/bigtea-probe --quick          # RAM/disk/GPU + what to close
./target/release/bigtea-model-info <m.gguf> --budget 8   # fit + tok/s prediction
```

Windows: needs the **GNU** Rust toolchain (`rustup default stable-x86_64-pc-windows-gnu`) plus MSYS2 mingw64 on PATH. `[[bin]]` targets set `test = false` — empty harnesses are pointless and Smart App Control blocks unsigned fresh binaries.

## Crates

`gguf` container parsing · `probe` hardware + RAM reclaim · `plan` prediction + residency policy · `io` cache-bypassing aligned reads · `model` sharded resolution + partial reads · `ggml` FFI (graph, zero-copy weight binding) · `tokenizer` byte-level BPE · `arch` architectures + streaming forward pass

## Facts that cost time to rediscover

- **ggml aborts** (`GGML_ASSERT`) when its arena is exhausted — no error to catch. Size arenas up front.
- **ggml `ne[0]` is the fastest dimension.** Reading shapes as row-major transposes every matrix and yields confident nonsense.
- **Weights are bound zero-copy** (`no_alloc` + data pointer). A copy would need 2× the model and not fit.
- **Missing causal mask → repeated tokens**, not an error. Masked positions need `-inf`, not `0`.
- **top_k does not return indices in score order** — look expert weights up by index.
- **Router weights must be renormalised** over selected experts only.
- A **wrong tokenizer or forward pass produces fluent nonsense**, never a crash. Test pieces separately.
- **`compute(&t, 0)` runs on ONE thread** — the count is floored at 1, not defaulted to all cores. This silently ran every expert matmul single-threaded.
- **Expert access is a cyclic scan, so recency-based caching is the worst policy available.** Layer 0 is always the oldest entry when layer 47 needs room. Frequency-gated admission took hit rate 17% → 70% at the same budget.
- **Profile before optimising a streaming runner.** The largest cost in generation was memcpy — slices copied twice per use — not disk and not arithmetic. Nothing suggested it until it was timed.

## Working rules

- Git: remote `github.com/aturzone/Bigtea`. Push with the token from `C:\Projects\.env` inline in the URL, output redacted — never in git config, never echoed. Model/weight files stay gitignored.
- Implementation goes on `ticket/<name>` branches + PR; Atur merges. Docs may go to main.
- Sync audit at phase boundaries only, not per commit.
- **A competitive claim is not citable until the competitor's exact command line and its output are in a doc.** "llama.cpp can't do X" survived days on a misattributed error string because nobody ran the opposing command. Run it, paste it, flag it.
- Keep this file under ~2000 tokens; tell Atur to prune rather than letting it bloat.

## Next

1. **DeepSeek-V4-Flash architecture — now the critical path, not a nice-to-have.** Model ≫ RAM is the only regime where this design should win, and we cannot test the thesis until Bigtea runs a model that large. llama.cpp does it at 0.45 tok/s with `--no-repack`; that is the bar. 1,203-line graph + bespoke KV cache; the 144 GB model is on disk.
2. Choose I/O mode from the model-size-to-RAM ratio. Bypassing the page cache is right when the model dwarfs RAM and wrong when it nearly fits — there we double-buffer against a kernel that uses all free RAM elastically.
3. Close the generation gap (we are ~2.3x behind). At 565 tokens: 3.2s disk, 5.4s expert compute, 1.0s attention, ~3.3s unattributed. Expert compute is single-column Q4_K matmuls; llama.cpp repacks for exactly this and we do not.
4. Headroom is 4 GiB, so we use 7.2 GiB where llama.cpp effectively gets ~11. Measure how much of the gap that alone explains.

## Compact Instructions

If auto-compacted, preserve ONLY: open decisions, the work in progress, files modified this session, unresolved questions for Atur. Discard tool output, committed file contents, and dead ends.
