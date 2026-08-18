# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, anything may change in a minor release.

## [Unreleased]

Nothing yet.

## [0.0.4] — 2026-08-18

### Added — one file to install everything

**`Chaos-0.0.4-Setup.exe`.** Download it, run it, press INSTALL. 23 MB with every
binary inside: no archive to unpack, no PowerShell, no toolchain, no network, no
administrator rights. It installs per-user, puts Chaos on your PATH, creates the
models folder, adds a Start Menu entry and registers in Add/Remove Programs.

Built without NSIS, WiX, Inno or MSI tooling, because every one of them would
have to be installed on the build machine before a release could be cut, and this
project has no dependencies. A Windows install turns out to be a window, a file
copy, a PATH entry, a shortcut and one registry key.

**Uninstalling never touches your models.** They live outside the install folder
on purpose, and a test enforces it — the failure mode is deleting a 155 GB
download.

Silent mode for scripting and CI: `/S`, `/S --uninstall`, `--prefix <dir>`.

### Added — Chaos as a window

**`chaos-app`**, a native Win32 application. Not a browser in a frame: a real
window, drawn with GDI, in two colours and nothing between them. Pick a model,
LOAD it, chat with it, UNLOAD it — and unloading genuinely frees the memory,
because the engine runs as a child process rather than inside the window.

INSTALLED and AVAILABLE tabs, a DOWNLOAD button, and settings for cache, threads
and port.

### Added — a browser interface for `chaos-serve`

`GET /` now serves a chat page, self-contained in the binary: no CDN, no font, no
script fetched from anywhere. An offline machine gets the whole interface.

### Added — 13 models to fetch, up from 2

Qwen3 4B/8B/14B/32B, Gemma-3 4B/12B/27B, Llama-3.2 1B/3B, Qwen2.5-Coder-7B and
Phi-4, alongside DeepSeek-V4-Flash and Qwen3-30B-A3B. Every repository, filename
and byte count was read from the Hugging Face API and verified to resolve before
being added.

**Each entry states what must stay resident, not just what it downloads**, and
that is the number the fit verdict uses:

```
v4flash    155.1 GB    7.38 GiB resident   -> streams on a 16 GB machine
qwen3-32b   19.8 GB   18.40 GiB resident   -> does not
```

A dense model has no routed experts, so nothing streams and the whole file has to
fit. Sorting by download size would have called the 155 GB model impossible and
the 20 GB one easy, which is backwards.

### Changed

- `chaos-iobench` and `chaos-gpubench` now ship in the release, so the
  measurements this project publishes can be reproduced by anyone who downloads
  it.
- The release workflow builds and tests the installer, and fails if it embedded
  nothing.

### Fixed

- The documented test count is now checked by CI against the suite that actually
  ran. It had gone stale three times — 566, 570, 575 — each caught only by
  someone noticing.


## [0.0.3] — 2026-08-16

### Changed — the project is now called `chaos`

`bigtea-run` is `chaos-run`, and so are the other ten binaries. Crate names,
`--help` text, info lines, environment variables (`BIGTEA_*` → `CHAOS_*`), the
workflows and all 105 documents were renamed in one pass — 1,623 occurrences
across 184 files, none left. The git remote is deliberately untouched; the
`repository`/`homepage` URLs point at the new name and start resolving when the
repository is renamed.

**This is a breaking change for anything scripted against the old names**, which
is what a `0.0.x` minor is for.

### Added — running it no longer starts with a path

**`chaos-run <name>`.** Every command used to begin with an absolute path to a
`.gguf` file, which on Windows means something like
`C:\Users\you\.chaos\models\Qwen3-30B-A3B-Q4_K_M.gguf` typed by hand, and for a
five-shard container it means knowing which shard to name. Now any unique part
of a name resolves:

```
chaos-run                                   # lists the models you have
chaos-run qwen3 "The capital of France is"  # runs Qwen3-30B-A3B-Q4_K_M.gguf
chaos-run deepseek "..."                    # opens shard 1 of 5, automatically
```

An existing path still wins, so nothing that worked before changes. An ambiguous
name lists the candidates rather than guessing a 144 GB read, and an unknown one
lists what *is* available rather than leaving the user to go looking.
`chaos-serve` has the same lookup from the same code, so the two cannot disagree
about where models live.

Searched in order: `CHAOS_MODELS`, `~/.chaos/models` (which `install.ps1`
creates), the download cache `chaos-pull` writes to, and `./models`. **Two of
those already existed and pointed at different places** — where a model lived
depended on how it had arrived — which is exactly the kind of thing a first-time
user should never have to learn.

### Added

- **A startup logo**: the name, then the logo centred beneath it, then the
  version. Rasterised offline into 3 KB of committed luminance bytes and printed
  with Unicode half-blocks, two pixels to a cell. Cropped to the artwork rather
  than the SVG's canvas, which had been carrying a wide white margin into every
  render. No SVG parser, no
  image decoder, no build script, no dependency — the workspace still has zero.
  It sizes itself to the terminal and is skipped for `NO_COLOR`,
  `CHAOS_NO_BANNER`, `--log-disable`, a terminal too small, and any stdout or
  stderr that is not a terminal.
- **`scripts/install.ps1`** — Windows install and in-place upgrade. Copies the
  binaries to `%LOCALAPPDATA%\Chaos\bin`, adds it to the *user* PATH exactly
  once, and creates `%USERPROFILE%\.chaos\models`. Re-running upgrades and
  removes binaries the new version no longer ships; `-Uninstall` reverses it and
  never touches the models directory. It refuses rather than half-upgrading when
  a binary is running. Shipped inside the Windows archive and smoke-tested in
  the release workflow, on the unpacked archive, in the shape a user meets it.
- `chaos-serve` and `chaos-pull` are now in the release archives. They were
  built and never packaged.

### Corrected

**V4-Flash is at parity with llama.cpp, not far behind it.** The published
figures — prefill 1.62x behind, generation 3-4x behind — date from 2026-08-07 and
no longer reproduce. Three alternating pairs in one session, both engines at
their defaults:

| DeepSeek-V4-Flash | Chaos | llama.cpp |
|---|---:|---:|
| prefill, ms per prompt token | **1640** | 1679 |
| generation, tok/s | **0.394** | 0.39 |

The warm-up run, discarded, read llama.cpp at 0.23 tok/s — which would have made
this a 1.7x lead. It is not one. See
[`where-we-stand-vs-llamacpp-2026-08-16.md`](docs/graph/research/where-we-stand-vs-llamacpp-2026-08-16.md).

**The parallel-experts optimisation does not transfer to V4-Flash, and the
ceiling is measured: the entire routed expert arithmetic is under 5% of a
token.** A token is 67% expert-slice read, 17% block compute, 16% routing. The
block's single `compute` had been folded into the residual of the phase table,
which is why that split had never been written down.

### Corrected — earlier

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

- `CHAOS_ROUTING_DUMP=<path>` writes raw `pass,layer,expert,count` rows, so two
  runs can be compared offline and passes are not conflated.
- `tools/routing/` — the prompts, capture script and analysis behind the above.
- `STATUS.md` — one canonical statement of where the project stands and what
  remains, so any session can resume without reconstructing it.

### Planned

Everything the previous list named is done — the KV cache, the downloader, the
OpenAI-compatible server, quant selection from the probe and prebuilt binaries.
What replaces it, in the order the measurements justify:

- **The tok/s-versus-RAM frontier for a 144 GB model.** Nobody has published it,
  and only an engine that owns residency can sweep it — `mmap` cannot be told to
  use exactly N GiB.
- **Verify the GPU tier.** `--device`, `-ngl`, `-ot` and `--op-offload` all work
  on Vulkan, and the device path fails 1 of 8 parity prompts where the CPU path
  fails none. Shown to be arithmetic rather than wiring, but unproven either way.
- **More architectures.** 13 of llama.cpp's 141 have been diffed against it.
- **Not** 20 tok/s on V4-Flash. That is closed by measurement rather than
  deferred: it needs 79 MB/token and the model reads 3288.

## [0.0.2] — 2026-08-07

Findings, a retraction, and the measurement that changes the project's direction.

> **⚠ Superseded 2026-08-08.** Every routing figure in this entry was scored
> in-sample on one prompt and four of them are wrong — see **Corrected** under
> [Unreleased](#unreleased). The entry is left as released rather than rewritten.

### Added

- `CHAOS_ROUTING=1` prints how often each expert of each layer is actually
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

**v0.0.1's claim that Chaos leads llama.cpp on DeepSeek-V4-Flash.** It claimed
3.0x faster load and 1.20x faster prefill. Both were false: Chaos's numbers were
measured fresh and llama.cpp's were copied from a two-day-old document taken under
different free-RAM conditions, so the engines were never run back to back. Run
back to back, twice:

| | Chaos | llama.cpp |
|---|---:|---:|
| load | 10.0s | 10.5s |
| prefill, per prompt token | 2440 ms | **1503 ms** |
| generation | 0.064 tok/s | **0.21–0.31 tok/s** |

**Chaos leads on nothing on this model.** It remains ahead on Qwen3-30B-A3B
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
- `CHAOS_THREADS` selects the thread count per graph evaluation, and
  `CHAOS_BLOCK_TIMING` now reports each phase of a block separately.

### Performance

DeepSeek-V4-Flash, same machine, both engines' command lines and outputs in
[`v4flash-vs-llamacpp-2026-08-07.md`](docs/graph/research/v4flash-vs-llamacpp-2026-08-07.md):

> **⚠ Retracted the same day.** This section originally claimed 3.0x faster load
> and 1.20x faster prefill. Both were wrong: Chaos's numbers were fresh and
> llama.cpp's were copied from a two-day-old document taken under different
> free-RAM conditions, so the two engines were never run back to back. Corrected
> figures, measured back to back twice:

| | Chaos | llama.cpp | |
|---|---:|---:|:--|
| load | 10.0s | 10.5s | parity |
| prefill, per prompt token | 2440 ms | **1503 ms** | llama.cpp 1.62x faster |
| generation | 0.064 tok/s | **0.21-0.31 tok/s** | llama.cpp 3-4x faster |

**Chaos leads on nothing on this model.** The speedups below are real and
measured against Chaos's own previous version; they simply did not close the gap.

A single-token forward pass costs **4.0s**. That is what one step of a KV-cached
loop will cost — 0.25 tok/s — and it is the number to plan against, because the
0.077 above is an artefact of re-running the whole sequence for each token.

### Fixed

- macOS: process enumeration read `/proc`, which does not exist there, so the
  "close these apps to free RAM" advice silently did nothing. Falls back to `ps`.
- macOS: Accelerate framework was never linked, though ggml's cmake enables it by
  default and calls vDSP.
- macOS: OpenMP was demanded unconditionally; AppleClang ships none.
  `CHAOS_GGML_OPENMP` overrides the per-platform default.
- The documented `cmake` line built **shared** ggml libraries, so a new user
  following the README got no `.a` archives at all.
- `chaos-arch` now fails with one actionable message when ggml is missing,
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
- `chaos-run` — prefill and generation.
- `chaos-probe` — RAM, disk, GPU, and what to close.
- `chaos-model-info` — fit prediction and tok/s estimate before running.
- `chaos-meta`, `gguf-info`, `chaos-loadbench` — container and I/O inspection.
- 157 unit tests and 16 container-backed tests.

### Performance

Measured on one machine (15.7 GiB RAM, NVMe at 2.55 GB/s, 20 threads). Both
engines produce identical, correct output; llama.cpp is measured with a warm page
cache. Full command lines and outputs in
[`head-to-head-llamacpp-2026-08-05.md`](docs/graph/research/head-to-head-llamacpp-2026-08-05.md).

Qwen3-30B-A3B Q4_K_M prefill, Chaos / llama.cpp:

| tokens | Chaos | llama.cpp |
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
  **Chaos leads on nothing** — see the retraction above. It is ahead only on
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

[Unreleased]: https://github.com/aturzone/Chaos/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/aturzone/Chaos/releases/tag/v0.0.2
[0.0.1]: https://github.com/aturzone/Chaos/releases/tag/v0.0.1
[0.0.0]: https://github.com/aturzone/Chaos/releases/tag/v0.0.0
