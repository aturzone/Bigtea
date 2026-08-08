# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, anything may change in a minor release.

## [Unreleased]

### Corrected

**The hot expert set is per-prompt, so it cannot be pinned.** v0.0.2's routing
figures were all scored *in-sample on a single prompt*. Re-measured on eight
prompts across four subjects, with the token-id-routed layers 0-2 excluded and a
uniform-router null at matched sample size:

| published in 0.0.2 | measured |
|---|---|
| top-64 = 97.8% of selections | **90.5%** in-sample, **53.7%** on a prompt the set was not chosen from |
| 33.6 tok/s disk floor at 34.27 GiB | **1.60 tok/s** |
| 20 tok/s needs a ~48 GiB desktop | unsupported — needs a 96.3% hit rate; a pinned cache gives 76.7% at 68.5 GiB |
| chi-square 7805 | not a valid statistic — generation re-runs prefill per token, so the prompt was counted once per pass (1282 → 5464 → 11469 for 1, 4, 8 passes, with coverage unmoved) |

The skew itself is real and reproduced on every prompt: top-8 of 256 takes
34.6–52.0% of selections against a uniform null of 6.8–7.4%. What does not hold
is *transfer* — across subjects a pinned hot set scores 37.5% against 25.0% for
caching at random. See
[`routing-skew-is-per-prompt-2026-08-08.md`](docs/graph/research/routing-skew-is-per-prompt-2026-08-08.md).

### Added

- `BIGTEA_ROUTING_DUMP=<path>` writes raw `pass,layer,expert,count` rows, so two
  runs can be compared offline and passes are not conflated.
- `tools/routing/` — the prompts, capture script and analysis behind the above.
- `STATUS.md` — one canonical statement of where the project stands and what
  remains, so any session can resume without reconstructing it.

### Planned

In the order the measurements justify — see
[`docs/graph/backlog/lts-0-0-0.md`](docs/graph/backlog/lts-0-0-0.md):

- KV cache for the DeepSeek-V4-Flash path — the only thing between us and a real
  generation number. A single-token pass already costs 4.0s; today every
  generated token re-runs the whole sequence instead.
- Overlap expert reads with compute. 2.3s of I/O and 1.0s of compute per token
  run serially; llama.cpp gets this overlap free from `mmap`. Layers 0-2 route by
  token id, so their expert set is knowable before any compute runs.
- Model downloader (`bigtea pull`) resolving names to Hugging Face repos, with
  resume, checksums and a disk-space check before starting.
- Quant selection from the hardware probe, with the tok/s prediction stated
  *before* a 144 GB download begins.
- OpenAI-compatible `/v1/chat/completions` server.
- Prebuilt binaries for Linux, macOS and Windows.

## [0.0.2] — 2026-08-07

Findings, a retraction, and the measurement that changes the project's direction.

> **⚠ Superseded 2026-08-08.** Every routing figure in this entry was scored
> in-sample on one prompt and four of them are wrong — see **Corrected** under
> [Unreleased](#unreleased). The entry is left as released rather than rewritten.

### Added

- `BIGTEA_ROUTING=1` prints how often each expert of each layer is actually
  selected, and what the hot set would cost to keep resident.

### Discovered

**DeepSeek-V4-Flash's router is violently skewed.** Every speed estimate this
project ever made assumed it spread evenly over 256 experts:

| top-N per layer | share of selections | resident cost |
|---:|---:|---:|
| 1 | 12.1% | 0.54 GiB |
| 8 | 52.9% | 4.28 GiB |
| 16 | 70.4% | 8.57 GiB |
| 64 | **97.8%** | 34.27 GiB |

Uniform routing would give top-16 = 6.2%; measured 70.4%, chi-square 7805 against
uniform's ~255. With a hot-set cache, bytes read per token fall from 3.21 GiB to
**72 MiB** — a 33.6 tok/s disk floor, against a 27 tok/s compute floor.

**20 tok/s for a 144 GB model is a cache-sizing problem, not a physics
violation**, and it needs roughly a **48 GiB desktop** rather than the ~150 GiB
previously claimed. On a 15.7 GiB laptop the same arithmetic implies ~1.3 tok/s,
about 4x llama.cpp. Neither is measured yet; both are arithmetic on measurements
that are. See
[`routing-skew-changes-everything.md`](docs/graph/research/routing-skew-changes-everything.md).

### Retracted

**v0.0.1's claim that Bigtea leads llama.cpp on DeepSeek-V4-Flash.** It claimed
3.0x faster load and 1.20x faster prefill. Both were false: Bigtea's numbers were
measured fresh and llama.cpp's were copied from a two-day-old document taken under
different free-RAM conditions, so the engines were never run back to back. Run
back to back, twice:

| | Bigtea | llama.cpp |
|---|---:|---:|
| load | 10.0s | 10.5s |
| prefill, per prompt token | 2440 ms | **1503 ms** |
| generation | 0.064 tok/s | **0.21–0.31 tok/s** |

**Bigtea leads on nothing on this model.** It remains ahead on Qwen3-30B-A3B
prefill at 565 and 2206 tokens, measured back to back.

## [0.0.1] — 2026-08-07

Performance. DeepSeek-V4-Flash prefill is **2.2x** faster than v0.0.0 and
generation **1.83x**, with every one of the 14 oracle tests still matching
llama.cpp's element sums.

### Changed

- **One graph evaluation per block instead of 24.** `Context::compute` evaluates
  a tensor's *entire ancestor graph*, so calling it on every intermediate does
  not merely dispatch more work — it **re-does** the work, once per call, and
  pays a graph build and a threadpool cycle each time. A value is now computed
  only where the CPU must read it. Worth **1.9x**, and invisible on a long
  prefill because the matmuls there are large enough to bury it.
- **A layer's three expert tensors are read in one parallel batch.** Four
  readers, jobs distributed one slice at a time so each reader gets an equal
  share of the bytes. Parallel reads had been tried and reverted twice before;
  the difference is batch size — per-tensor groups are 6 slices at generation
  time, and the thread spawns cost more than the queue depth buys.
- `BIGTEA_THREADS` selects the thread count per graph evaluation, and
  `BIGTEA_BLOCK_TIMING` now reports each phase of a block separately.

### Performance

DeepSeek-V4-Flash, same machine, both engines' command lines and outputs in
[`v4flash-vs-llamacpp-2026-08-07.md`](docs/graph/research/v4flash-vs-llamacpp-2026-08-07.md):

> **⚠ Retracted the same day.** This section originally claimed 3.0x faster load
> and 1.20x faster prefill. Both were wrong: Bigtea's numbers were fresh and
> llama.cpp's were copied from a two-day-old document taken under different
> free-RAM conditions, so the two engines were never run back to back. Corrected
> figures, measured back to back twice:

| | Bigtea | llama.cpp | |
|---|---:|---:|:--|
| load | 10.0s | 10.5s | parity |
| prefill, per prompt token | 2440 ms | **1503 ms** | llama.cpp 1.62x faster |
| generation | 0.064 tok/s | **0.21-0.31 tok/s** | llama.cpp 3-4x faster |

**Bigtea leads on nothing on this model.** The speedups below are real and
measured against Bigtea's own previous version; they simply did not close the gap.

A single-token forward pass costs **4.0s**. That is what one step of a KV-cached
loop will cost — 0.25 tok/s — and it is the number to plan against, because the
0.077 above is an artefact of re-running the whole sequence for each token.

### Fixed

- macOS: process enumeration read `/proc`, which does not exist there, so the
  "close these apps to free RAM" advice silently did nothing. Falls back to `ps`.
- macOS: Accelerate framework was never linked, though ggml's cmake enables it by
  default and calls vDSP.
- macOS: OpenMP was demanded unconditionally; AppleClang ships none.
  `BIGTEA_GGML_OPENMP` overrides the per-platform default.
- The documented `cmake` line built **shared** ggml libraries, so a new user
  following the README got no `.a` archives at all.
- `bigtea-arch` now fails with one actionable message when ggml is missing,
  instead of a wall of unresolved imports.
- Declared MSRV was 1.74 while the code used a 1.82 API. Now 1.82.

## [0.0.0] — 2026-08-07

First public release. The engine works and is verified; the product around it is
not built yet. See [README](README.md#status) for what is and is not there.

### Added

- **Runs Mixture-of-Experts models several times larger than RAM, on CPU.**
  Always-read weights stay resident; routed experts stream from disk as routing
  selects them.
- **DeepSeek-V4-Flash (`deepseek4`) support** — 43 blocks, hyper-connections,
  three kinds of compressed attention (raw, compressed-sparse, heavily
  compressed), hash routing on the first three layers and biased top-k routing on
  the other 40. Verified element-by-element against llama.cpp on all 43 blocks
  plus the output head.
- **Qwen3 and Qwen3-MoE support**, with a frequency-gated expert cache.
- **Cache-bypassing direct I/O** (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`), falling
  back to buffered reads and *reporting* that it did rather than pretending.
- **Zero-copy expert reads.** `SkewedBuf` deliberately misaligns the destination
  buffer to match the file's sector residue, because GGUF pads tensor data to 32
  bytes rather than to a disk sector — so a conventionally aligned buffer can
  never receive a direct transfer. 0.80 → 1.58 GiB/s, with 0.09% of bytes copied
  instead of 300%.
- **Residency with a hard budget**, which reports what did not fit, what
  re-reading it costs per token, and which processes to close to fix it.
- `bigtea-run` — prefill and generation.
- `bigtea-probe` — RAM, disk, GPU, and what to close.
- `bigtea-model-info` — fit prediction and tok/s estimate before running.
- `bigtea-meta`, `gguf-info`, `bigtea-loadbench` — container and I/O inspection.
- 157 unit tests and 16 container-backed tests.

### Performance

Measured on one machine (15.7 GiB RAM, NVMe at 2.55 GB/s, 20 threads). Both
engines produce identical, correct output; llama.cpp is measured with a warm page
cache. Full command lines and outputs in
[`head-to-head-llamacpp-2026-08-05.md`](docs/graph/research/head-to-head-llamacpp-2026-08-05.md).

Qwen3-30B-A3B Q4_K_M prefill, Bigtea / llama.cpp:

| tokens | Bigtea | llama.cpp |
|---:|---:|---:|
| 565 | **27.64** | 23.55 |
| 2206 | **36.60** | 33.59 |
| 4395 | 38.40 | 40.25 |
| 8775 | 34.88 | 35.01 |
| 4395 (`-b 4096`) | **43.61** | 40.25 |

### Known limitations

- **Generation is slower than llama.cpp.** DeepSeek-V4-Flash: 0.077 tok/s against
  0.45, because the V4-Flash path has no KV cache yet and each token re-runs the
  whole sequence. Qwen3-30B-A3B: 1.07 against 2.16, about 2x. On V4-Flash
  **Bigtea leads on nothing** — see the retraction above. It is ahead only on
  Qwen3-30B-A3B prefill at 565 and 2206 tokens.
- **Linux and macOS build and pass the unit tests in CI, but no model has been
  run on either.** macOS additionally has no direct-I/O path — `F_NOCACHE` needs
  an `fcntl` after opening and is not written yet — so it falls back to buffered
  reads and the page-cache problems this design exists to avoid.
- No model downloader; bring your own `.gguf`.
- No server or API.
- No prebuilt binaries; ggml must be built first and `GGML_LIB_DIR` set.
- Only `qwen3`, `qwen3moe` and `deepseek4` architectures run. Others parse as
  containers but will not execute.

### Retracted

- **"llama.cpp cannot run models larger than RAM."** It can, with `--no-repack`.
  This claim survived several days on a misattributed error string because nobody
  ran the opposing command. It is retracted in writing, and the project now
  requires a competitor's exact command line and output before any competitive
  claim is citable.

[Unreleased]: https://github.com/aturzone/Bigtea/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/aturzone/Bigtea/releases/tag/v0.0.2
[0.0.1]: https://github.com/aturzone/Bigtea/releases/tag/v0.0.1
[0.0.0]: https://github.com/aturzone/Bigtea/releases/tag/v0.0.0
