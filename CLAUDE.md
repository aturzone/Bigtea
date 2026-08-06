# Bigtea — a runner for models larger than RAM

- **What it is**: a Rust inference runner whose job is running models that do *not* fit in memory. Keeps the always-read weights resident, streams routed experts from disk per token. Borrows `ggml` for arithmetic; owns memory, residency, streaming, and the token loop.
- **Proven**: Qwen3-30B-A3B (17.28 GiB container) generates correct text on a 15.7 GiB machine holding 0.93 GiB resident + a 6.26 GiB expert cache.
- **Prefill beats llama.cpp** at 565 (27.6 vs 23.6) and 2206 tokens (36.6 vs 33.6), and matches it at 4395 and 8775; `-b 4096` gives 43.6 vs 40.3. **Generation is still ~2x behind** (1.07 vs 2.16) — do not claim otherwise. llama.cpp also runs the 144 GB V4-Flash once `--no-repack` is passed, so "larger than RAM" is not a differentiator. Full ladder, retracted claims, and one experiment that failed: `docs/graph/research/head-to-head-llamacpp-2026-08-05.md`.
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
- **Cache hit rate is not a success metric.** Past ~6 GiB the expert cache reaches 71% hits and is the *slowest* configuration measured: cached bytes get paged out, so a "hit" is a page fault wearing a disguise. Only tok/s at a stated footprint counts.
- **`flash_attn_ext` does NOT transpose V**, unlike the `mul_mat` attention path, and its mask must be **F16 and contiguous**. Both mistakes give fluent nonsense, not an error. Mask values are only 0 and -inf, so write the bits (`0x0000` / `0xFC00`) rather than converting.
- **Every arena must scale with the prefill block.** Fixed-size arenas abort once the block grows; ggml asks and dies rather than returning an error.

## Working rules

- Git: remote `github.com/aturzone/Bigtea`. Push with the token from `C:\Projects\.env` inline in the URL, output redacted — never in git config, never echoed. Model/weight files stay gitignored.
- Implementation goes on `ticket/<name>` branches + PR; Atur merges. Docs may go to main.
- Sync audit at phase boundaries only, not per commit.
- **A competitive claim is not citable until the competitor's exact command line and its output are in a doc.** "llama.cpp can't do X" survived days on a misattributed error string because nobody ran the opposing command. Run it, paste it, flag it.
- Keep this file under ~2000 tokens; tell Atur to prune rather than letting it bloat.

## Next

1. **DeepSeek-V4-Flash — the critical path.** Its 7.38 GiB of always-read weights *fit* this machine, with 137.06 GiB of experts streamed (3.21 GiB/token, 6 of 256). That is the regime the design targets and the only place it should beat llama.cpp, whose dense weights get evicted by cold expert traffic. Bar: 0.45 tok/s. Physics ceiling: ~0.87 tok/s cold. **Forward pass verified end to end** — all 43 blocks + head match llama.cpp and emit a sane token, but only at a 2-token prompt, which is where the compressed attention builders stay switched off. **Next: the compressors and lightning indexer, which 41 of 43 blocks need at any real prompt length.** `v4flash-port-recon.md`, `v4flash-compressed-attention.md`.
2. Choose I/O mode from the model-size-to-RAM ratio. Bypassing the page cache is right when the model dwarfs RAM and wrong when it nearly fits — there we double-buffer against a kernel that uses all free RAM elastically.
3. **Close the generation gap — the only place we still lose (~2x). Repack Q4_K expert slices on cache admission.** Expert compute is 60% of generation, and it is neither barrier-bound (12→4 threads costs nothing) nor bandwidth-bound (2.4 GB/s against DDR5) — it is dequantisation. llama.cpp interleaves rows so several unpack per SIMD op (`REPACK = 1`); repacking once when a slice enters the cache fits this design well, since it is then reused by every token that routes there.
4. Auto-tune the prefill block from free RAM. Block size is worth more than any kernel here (512 → 4096 is 30.5 → 43.6 tok/s) and it is currently a fixed 2048 with a `-b` override.

## Compact Instructions

If auto-compacted, preserve ONLY: open decisions, the work in progress, files modified this session, unresolved questions for Atur. Discard tool output, committed file contents, and dead ends.
