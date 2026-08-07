# Bigtea — a runner for models larger than RAM

- **What it is**: a Rust inference runner whose job is running models that do *not* fit in memory. Keeps the always-read weights resident, streams routed experts from disk per token. Borrows `ggml` for arithmetic; owns memory, residency, streaming, and the token loop.
- **Proven**: Qwen3-30B-A3B (17.28 GiB container) generates correct text on a 15.7 GiB machine holding 0.93 GiB resident + a 6.26 GiB expert cache.
- **Prefill beats llama.cpp** at 565 (27.6 vs 23.6) and 2206 tokens (36.6 vs 33.6), and matches it at 4395 and 8775; `-b 4096` gives 43.6 vs 40.3. **Generation is still ~2x behind** (1.07 vs 2.16) — do not claim otherwise. llama.cpp also runs the 144 GB V4-Flash once `--no-repack` is passed, so "larger than RAM" is not a differentiator. Full ladder, retracted claims, and one experiment that failed: `docs/graph/research/head-to-head-llamacpp-2026-08-05.md`.
- Graph docs live in `/docs/graph/`; read `INDEX.md` first, then only the 2–3 nodes a task links to.

## Build / test / run

```
# ggml must be built first; point GGML_LIB_DIR at ggml-base.a, ggml-cpu.a, ggml.a
export GGML_LIB_DIR=C:/Projects/llamacpp-unsloth/build/ggml/src   # PowerShell: $env:GGML_LIB_DIR=...
cargo test --release          # 153 tests (+12 container-backed, --ignored)
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
- **Cache hit rate is not a success metric.** Past ~6 GiB the expert cache reaches 71% hits and is the *slowest* configuration measured: cached bytes get paged out, so a "hit" is a page fault wearing a disguise. Only tok/s at a stated footprint counts.
- **`flash_attn_ext` does NOT transpose V**, unlike the `mul_mat` attention path, and its mask must be **F16 and contiguous**. Both mistakes give fluent nonsense, not an error. Mask values are only 0 and -inf, so write the bits (`0x0000` / `0xFC00`) rather than converting.
- **Prompt length decides which code paths run.** V4-Flash's compressed attention builders are guarded on their caches being non-empty, so the *same layer* runs different attention at different lengths: at 2 tokens all 43 blocks fall back to the Raw path, at 5 CSA fires, at 165 HCA fires, and the sparse indexer selects nothing until >2048. A shorter capture can reach *further* than a longer one. See `v4flash-compressed-attention.md`.
- **GGUF pads tensor data to `general.alignment` (32), not to a disk sector.** So tensors start mid-sector and a conventionally *aligned* buffer can never receive a direct transfer — every byte bounces. Skew the destination to `file_offset % 4096` instead (`SkewedBuf`): 0.80 → 1.58 GiB/s, 0.09% copied.
- **`compute()` re-evaluates the whole ancestor graph.** Calling it per intermediate *re-does* the work each time, plus a graph build and threadpool cycle. 24 calls per block became 6 — **1.9x**. Invisible on prefill (big matmuls bury it), dominant at one token. Compute only before a `to_vec_*`/`set_*`.
- **Threads are not the lever.** 4/12/20 threads all cost the same on a V4-Flash prefill; 1 thread is 4.7x *slower*. Threadpool-churn was the obvious explanation and was wrong.
- **Every arena must scale with the prefill block.** Fixed-size arenas abort once the block grows; ggml asks and dies rather than returning an error.

## Working rules

- Git: remote `github.com/aturzone/Bigtea`. Push with the token from `C:\Projects\.env` inline in the URL, output redacted — never in git config, never echoed. Model/weight files stay gitignored.
- Implementation goes on `ticket/<name>` branches + PR; Atur merges. Docs may go to main.
- Sync audit at phase boundaries only, not per commit.
- **A competitive claim is not citable until the competitor's exact command line and its output are in a doc.** "llama.cpp can't do X" survived days on a misattributed error string because nobody ran the opposing command. Run it, paste it, flag it.
- Keep this file under ~2000 tokens; tell Atur to prune rather than letting it bloat.

## Next

**v0.0.0 released 2026-08-07**, CI green on Linux/macOS/Windows. Full head-to-head: `v4flash-vs-llamacpp-2026-08-07.md`.

**V4-Flash vs llama.cpp today**: load **3.0x ahead** (4.1s vs 12.3s), prefill **1.20x ahead** (0.49 vs 0.41 tok/s), generation **5.8x behind** (0.077 vs 0.45).

1. **KV cache** — the only thing between us and a real generation number. A single-token pass costs **4.0s**, which is what a cached step will cost (0.25 tok/s); today each token re-runs the whole sequence. State is bounded: 128 raw positions + 256 compressed blocks ≈ 33 MB for 43 layers. Needs an oracle capture at two consecutive positions — a wrong cache gives fluent nonsense.
2. **Overlap reads with compute.** 2.3s I/O vs 1.0s compute per token, run serially. llama.cpp gets this free from mmap. Layers 0-2 route by token id so their experts are knowable before any compute. Worth ~1.0s/token.
3. **Fit the always-read set** — 1.17 GiB short on this machine. Worth 0.7s/token, and it is the user's RAM, not code.
4. Then T1-T5 of `lts-0-0-0.md`: `bigtea pull` from Hugging Face, quant selection from the probe, self-configuration, OpenAI-compatible server, prebuilt binaries.

Levers 1-3 reach **~0.43 tok/s = parity**, not victory. Beating 0.45 needs an expert cache across tokens, unmeasured on this model.

## Compact Instructions

If auto-compacted, preserve ONLY: open decisions, the work in progress, files modified this session, unresolved questions for Atur. Discard tool output, committed file contents, and dead ends.
