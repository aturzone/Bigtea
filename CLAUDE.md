# Bigtea — a runner for models larger than RAM

> **Read `STATUS.md` first.** It is the canonical statement of where the project
> is, the honest scoreboard, and what remains in order. Update it in the same
> commit as anything that moves a number or closes a task.

- **What it is**: a Rust inference runner whose job is running models that do *not* fit in memory. Keeps the always-read weights resident, streams routed experts from disk per token. Borrows `ggml` for arithmetic; owns memory, residency, streaming, and the token loop.
- **Proven**: Qwen3-30B-A3B (17.28 GiB container) generates correct text on a 15.7 GiB machine holding 0.93 GiB resident + a 6.26 GiB expert cache.
- **Prefill beats llama.cpp** at 565 (27.6 vs 23.6) and 2206 tokens (36.6 vs 33.6), and matches it at 4395 and 8775; `-b 4096` gives 43.6 vs 40.3. **Generation is still ~2x behind** (1.07 vs 2.16) — do not claim otherwise. llama.cpp also runs the 144 GB V4-Flash once `--no-repack` is passed, so "larger than RAM" is not a differentiator. Full ladder, retracted claims, and one experiment that failed: `docs/graph/research/head-to-head-llamacpp-2026-08-05.md`.
- Graph docs live in `/docs/graph/`; read `INDEX.md` first, then only the 2–3 nodes a task links to.

## Build / test / run

```
# ggml must be built first; point GGML_LIB_DIR at ggml-base.a, ggml-cpu.a, ggml.a
export GGML_LIB_DIR=C:/Projects/llamacpp-unsloth/build/ggml/src   # PowerShell: $env:GGML_LIB_DIR=...
cargo test --release          # 224 tests
cargo test --release --test deepseek4_forward -- --ignored   # 19 V4-Flash, needs the container
cargo build --release
./target/release/bigtea-run <model.gguf> "prompt" -n 16
./target/release/bigtea-probe --quick          # RAM/disk/GPU + what to close
./target/release/bigtea-model-info <m.gguf> --budget 8   # fit + tok/s prediction
```

Windows: needs the **GNU** Rust toolchain (`rustup default stable-x86_64-pc-windows-gnu`) plus MSYS2 mingw64 on PATH. `[[bin]]` targets set `test = false` — empty harnesses are pointless and Smart App Control blocks unsigned fresh binaries.

## Crates

`gguf` container parsing · `probe` hardware + RAM reclaim · `plan` prediction + residency policy · `io` cache-bypassing aligned reads · `model` sharded resolution + partial reads · `ggml` FFI (graph, zero-copy weight binding) · `tokenizer` byte-level BPE · `arch` architectures + streaming forward pass

## Facts that cost time to rediscover

- **ggml aborts** (`GGML_ASSERT`) when its arena is exhausted — no error to catch. Size arenas up front. **This also kills a whole test binary**: the 19 V4-Flash tests each allocate GB-sized arenas and, run in parallel, exhausted memory and aborted the process — reported as `process didn't exit successfully`, not as a failing test, with every later result lost. They hold a shared `heavy()` lock now, so plain `--ignored` works.
- **ggml `ne[0]` is the fastest dimension.** Reading shapes as row-major transposes every matrix and yields confident nonsense.
- **Weights are bound zero-copy** (`no_alloc` + data pointer). A copy would need 2× the model and not fit.
- **Missing causal mask → repeated tokens**, not an error. Masked positions need `-inf`, not `0`.
- **top_k does not return indices in score order** — look expert weights up by index.
- **Router weights must be renormalised** over selected experts only.
- A **wrong tokenizer or forward pass produces fluent nonsense**, never a crash. Test pieces separately.
- **`compute(&t, 0)` runs on ONE thread** — the count is floored at 1, not defaulted to all cores. This silently ran every expert matmul single-threaded.
- **Expert access is a cyclic scan, so recency-based caching is the worst policy available.** Layer 0 is always the oldest entry when layer 47 needs room. Frequency-gated admission took hit rate 17% → 70% at the same budget.
- **Profile before optimising a streaming runner.** The largest cost in generation was memcpy — slices copied twice per use — not disk and not arithmetic. Nothing suggested it until it was timed.
- **Expert reads are deduplicated per block across the whole batch.** A pass reads the *distinct* experts its tokens select, not one slice per selection (`read_expert_slices` takes `unique`). Measured distinct experts per layer per pass: **6 at one token (3.2 GiB), 39.7 at 17 tokens (21 GiB), 122.8 at 166 tokens (66 GiB)** — selections per layer grow 10x from 17 to 166 tokens while distinct reads only grow 3x. **So a cache's value depends on how many distinct experts a step touches, not on how skewed routing is**, and only a KV-cached single-token step is small enough for a few GiB to cover.
- **Routing is not bitwise stable across sequence lengths.** At 63 → 64 tokens the *same* earlier tokens re-routed ~3% of their selections (net still +6 per layer, so nothing was lost) — near-ties in the top-6-of-256 flipping when the batch shape changes. Layers 0-2 (token-id routed) were untouched, so it arrives through attention. **Mechanism unidentified**: "ggml re-blocks at multiples of 64" was the first guess and a 166→212 run crossing 192 showed zero churn, so it is not that. A test demanding equal routing across batch shapes will fail on correct code.
- **A hot set scored on the prompt it was chosen from tells you nothing.** "64 experts absorb 97.8% of selections" was in-sample on one prompt; out of sample it is 53.7%, and 37.5% across subjects against 25% for caching at random. Always score a residency policy on data it did not see. Two matching controls are cheap and both were missing: a **uniform null at the same sample size** (with ~1000 draws over 256 experts, top-64 covers 41% by construction) and a **noise ceiling** (resample the same distribution — if cross-prompt sits below it, the divergence is real).
- **Statistics computed over `bigtea-run`'s output double-count.** Regeneration is stateless, so every generated token re-runs prefill and the routing histogram counts the same prompt again: chi-square went 1282 → 5464 → 11469 for 1/4/8 tokens while coverage never moved. Capture with `-n 1`.
- **Cache hit rate is not a success metric.** Past ~6 GiB the expert cache reaches 71% hits and is the *slowest* configuration measured: cached bytes get paged out, so a "hit" is a page fault wearing a disguise. Only tok/s at a stated footprint counts.
- **`flash_attn_ext` does NOT transpose V**, unlike the `mul_mat` attention path, and its mask must be **F16 and contiguous**. Both mistakes give fluent nonsense, not an error. Mask values are only 0 and -inf, so write the bits (`0x0000` / `0xFC00`) rather than converting.
- **Prompt length decides which code paths run.** V4-Flash's compressed attention builders are guarded on their caches being non-empty, so the *same layer* runs different attention at different lengths: at 2 tokens all 43 blocks fall back to the Raw path, at 5 CSA fires, at 165 HCA fires, and the sparse indexer selects nothing until >2048. A shorter capture can reach *further* than a longer one. See `v4flash-compressed-attention.md`.
- **GGUF pads tensor data to `general.alignment` (32), not to a disk sector.** So tensors start mid-sector and a conventionally *aligned* buffer can never receive a direct transfer — every byte bounces. Skew the destination to `file_offset % 4096` instead (`SkewedBuf`): 0.80 → 1.58 GiB/s, 0.09% copied.
- **`compute()` re-evaluates the whole ancestor graph.** Calling it per intermediate *re-does* the work each time, plus a graph build and threadpool cycle. 24 calls per block became 6 — **1.9x**. Invisible on prefill (big matmuls bury it), dominant at one token. Compute only before a `to_vec_*`/`set_*`.
- **Threads are two levers pulling opposite ways, and `-t` reached only one architecture.** Generation saturates DRAM and wants **2-4** threads; prefill is compute-bound and wants **all** of them (Qwen3-4B: gen 7.64 @2 vs 4.49 @20; prefill 47.4 @4 vs 81.5 @20). Hence `-t` *and* `-tb`, picked by the step's token count. The old "threads are not the lever" reading came from a sweep whose knob was disconnected — `-t` set `BIGTEA_THREADS`, which only `deepseek4_forward.rs` read, so `-t 1` and `-t 20` gave *bit-identical* phase timings. **A disconnected knob is indistinguishable from a flat response; check the knob moves something first.** Fixing it was 1.66x/1.69x.
- **Do not calibrate on a proxy.** A 150 ms DRAM-saturation benchmark picked 6/8/12/12/4/6 on six identical runs while the true optimum was 2-4, and its spread was worse than the bad default it replaced — a pure read has no per-node barrier, a ggml graph does. Tune on real generated tokens instead. A proxy corrected until it agrees with the objective *is* the objective, measured badly.
- **Concurrent readers need a file handle EACH.** A Windows handle without `FILE_FLAG_OVERLAPPED` is synchronous and the OS serialises reads on it, so N threads on one handle hold the drive at queue depth 1. The old "no gain past 4 readers, the drive does 2.37 GiB/s" was this artefact: same reads, 4 threads, **2.01 GiB/s shared vs 2.65 per-handle**, and per-handle beats the "sequential ceiling". `Shard` now pools 8 handles.
- **The expert matmul is 3% of a token, not the floor.** 3.02 ms per block = 0.13 s/token, at 24.7 GiB/s — *above* single-threaded memcpy, i.e. already at DRAM speed. **76% of a token is disk.** Compute also scales as ~`n^0.49` in the batch, not linearly, so batched/speculative passes are cheaper than a linear model predicts.
- **Every arena must scale with the prefill block.** Fixed-size arenas abort once the block grows; ggml asks and dies rather than returning an error.
- **V4-Flash has no redundancy left to harvest — four probes, four negatives.** Experts are 9.1% internally negligible; the expert *bank* is full-rank (a rank-512 shared basis holds 20.4% of its energy against **16.6% for random noise**, `bigtea-spectrum`); the router's tail is not small (33.5/20.6/15.0/12.1/10.1/**8.8**%, so 3-of-6 discards 31% of the mass); and a pinned hot set scores 37.5% vs 25.0% random. **3.21 GiB/token is what the model costs, not an artefact.** Do not re-propose factorisation, contextual sparsity, or pinning.
- **Speculative decoding is ~1.4x here, not 2.2x.** The literature assumes the verify pass costs what a single-token pass costs; here it costs more, because more tokens select more distinct experts (`U(n)≈6·n^0.667`). Below α≈0.75 it is a net *loss*, and the optimum draft is short.
- **Windows: `.cargo/config.toml` sets `link-self-contained=no`.** MSYS2 gcc 16.1.0 dropped symbols rustup's bundled `crt2.o` still references, so every link fails with "undefined reference" on code that compiles. Do not delete it.

## Working rules

- Git: remote `github.com/aturzone/Bigtea`. Push with the token from `C:\Projects\.env` inline in the URL, output redacted — never in git config, never echoed. Model/weight files stay gitignored.
- Implementation goes on `ticket/<name>` branches + PR; Atur merges. Docs may go to main.
- Sync audit at phase boundaries only, not per commit.
- **A competitive claim is not citable until the competitor's exact command line and its output are in a doc.** "llama.cpp can't do X" survived days on a misattributed error string because nobody ran the opposing command. Run it, paste it, flag it.
- Keep this file under ~2000 tokens; tell Atur to prune rather than letting it bloat.

## Next

**v0.0.2 released 2026-08-07.** Read `backlog/next-session-handoff.md` first — it carries R0-R6 in measurement order.

**V4-Flash vs llama.cpp, run back to back**: load parity, prefill **1.62x behind**, generation **3-4x behind**. We lead on nothing on this model. Do not claim otherwise: `v4flash-vs-llamacpp-2026-08-07.md`.

**R0 done 2026-08-08** (`routing-skew-is-per-prompt-2026-08-08.md`): the router is genuinely skewed — top-8 takes 5-7x a uniform router — but **the hot set is per-prompt and must be warmed, not pinned.** Pinned from one prompt it covers 61.3% of another on the same subject, 37.5% across subjects, against 25.0% for a *random* cache. This corrected four v0.0.2 figures, killed the "prune the model to its hot set" plan, and reshaped R1.

**R0.1, R1, R3 done. R5 started** — `bigtea-pull`, `bigtea-serve` (OpenAI-compatible, verified against the live model), release workflow.

**2026-08-10** (`v4flash-has-no-slack-2026-08-10.md`): the byte-reduction roadmap is closed. 20 tok/s needs 79 MB/token; V4-Flash reads 3288. Everything still alive multiplies to **3.1x** against a **42x** gap, and the two ideas that could have closed it were measured and failed. **20 tok/s is not a code problem** — it needs the active weights to stop coming from disk.

1. **The tok/s-versus-RAM frontier for a 144 GB model** — never published by anyone, and only an engine that owns residency can sweep it (`mmap` cannot be told to use exactly N GiB). Answers the product question honestly: *given your machine, the largest model at the speed you want.*
2. **Overlap reads with compute** — ~53 ms read vs ~23 ms compute per block, ceiling ~1.4x.
3. **Ring wraparound** to lift the 256-token context ceiling (#46).
4. Finish R5/T1-T5 of `lts-0-0-0.md`: quant selection, self-configuration, prebuilt binaries.

**No GPU code exists** — `bigtea-probe` detects the card, nothing uses it. A VRAM tier needs a CUDA-enabled ggml *and* a non-zero-copy binding path, since weights are bound by handing ggml a host pointer.

## Compact Instructions

If auto-compacted, preserve ONLY: open decisions, the work in progress, files modified this session, unresolved questions for Atur. Discard tool output, committed file contents, and dead ends.
