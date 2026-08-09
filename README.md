<h1 align="center">Bigtea</h1>

<p align="center">
  <strong>A CPU inference runner for models that do not fit in RAM.</strong><br>
  Keeps the always-read weights resident, streams routed experts from disk per token.
</p>

<p align="center">
  <a href="#status"><img alt="status" src="https://img.shields.io/badge/status-v0.0.2%20preview-orange"></a>
  <a href="LICENSE"><img alt="licence" src="https://img.shields.io/badge/licence-Apache--2.0-blue"></a>
  <a href="#building"><img alt="rust" src="https://img.shields.io/badge/rust-1.82%2B-informational"></a>
  <a href="https://github.com/aturzone/Bigtea/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/aturzone/Bigtea/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="tests" src="https://img.shields.io/badge/tests-168%20passing-brightgreen">
</p>

---

> **[STATUS.md](STATUS.md)** is the current state of the project in one page —
> the honest scoreboard against llama.cpp, what works, the known limitations, and
> what is being worked on next. Read it before quoting any number from here.

Bigtea runs a **144 GB** Mixture-of-Experts model on a **15.7 GiB** laptop, on CPU,
and produces correct text. It does that by never loading the model: the weights
every token needs stay in RAM, and the routed experts — which are most of the
container and of which a token uses six of 256 — are read from disk as routing
selects them.

```console
$ bigtea-run DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf "The capital of France is" -n 5
model      deepseek4 (direct (cache bypassed))
shape      43 blocks, 4096 embd, 64 heads, 256 experts (6 used, 1 shared)
resident   251 tensors, 6.21 GiB of 6.21 GiB budget in 4.1s (1.65 GB/s)
           1.17 GiB will be re-read from disk on EVERY token
           closing these would free up to 5.63 GiB:
             Code.exe                     1.55 GiB (15 processes)
             Telegram.exe                 0.66 GiB
           that is enough to make the whole model resident.
prefill    5 tokens in 10.1s (0.49 tok/s)
output      Paris.
generate   4 tokens in 51.9s (0.077 tok/s, 13.0s per token)
```

---

## Read this before you try it

**Bigtea is slower than llama.cpp.** Measured back to back on the same machine,
DeepSeek-V4-Flash:

| | Bigtea | llama.cpp | |
|---|---:|---:|:--|
| load | 10.0s | 10.5s | parity |
| prefill, per prompt token | 2440 ms | **1503 ms** | llama.cpp **1.62x** faster |
| generation | 0.064 tok/s | **0.21-0.31** | llama.cpp **3-4x** faster |

Both command lines and both outputs:
[`v4flash-vs-llamacpp-2026-08-07.md`](docs/graph/research/v4flash-vs-llamacpp-2026-08-07.md).

> **Retraction.** v0.0.1 claimed Bigtea was 3.0x faster on load and 1.20x on
> prefill for this model. That was **wrong**: Bigtea's numbers were fresh and
> llama.cpp's were copied from a two-day-old document taken under different
> conditions. Run back to back, we lose on both. The claim is withdrawn.

Where Bigtea **is** ahead is Qwen3-30B-A3B prefill, measured back to back:

| prompt tokens | Bigtea | llama.cpp | |
|---:|---:|---:|:--|
| 565 | **27.64** | 23.55 | ahead |
| 2206 | **36.60** | 33.59 | ahead |
| 4395 | 38.40 | 40.25 | behind |
| 8775 | 34.88 | 35.01 | parity |

**llama.cpp also runs models larger than RAM**, given `--no-repack`. "Runs a model
bigger than your memory" is not a thing only Bigtea does; that claim is retracted
too, in
[`head-to-head-llamacpp-2026-08-05.md`](docs/graph/research/head-to-head-llamacpp-2026-08-05.md).

If you want the fastest local inference today, use
[llama.cpp](https://github.com/ggml-org/llama.cpp). Bigtea is worth your time if
you care about *how* larger-than-RAM inference behaves and want an engine that
measures and reports it rather than guessing.

## Why it might still interest you

- **It tells you what it is doing.** Resident bytes, what did not fit, what that
  costs per token, which processes to close to fix it, bytes read per layer,
  expert cache hit rate. Not a progress bar — numbers you can act on.
- **Every performance claim in this repository has a command line and its output
  behind it.** Claims that failed verification are kept, marked retracted, with
  the reason. See [Engineering notes](#engineering-notes).
- **The forward pass is verified element-by-element against llama.cpp.** All 43
  blocks of DeepSeek-V4-Flash plus the output head match llama.cpp's own element
  sums, across all three of its attention kinds. A wrong forward pass produces
  fluent nonsense, never a crash, so this is checked rather than assumed.

## Status

**v0.0.2 — preview.** The engine works and is verified. The product around it is
not built yet.

| | |
|---|---|
| ✅ Runs models several times larger than RAM, on CPU | Qwen3-30B-A3B (17.28 GiB) and DeepSeek-V4-Flash (144 GB) both on a 15.7 GiB machine |
| ✅ Correct output, verified against llama.cpp | 168 unit tests + 16 container-backed tests |
| ✅ Cache-bypassing direct I/O with zero-copy expert reads | |
| ✅ Honest reporting of residency, throughput and shortfalls | |
| ⚠️ **Generation is slow** | see above — no KV cache yet on the V4-Flash path |
| ⚠️ **V4-Flash is limited to 256 prompt tokens** | it builds its attention cache for the whole sequence at once; longer prompts are refused with a message. Lifting this is part of the KV-cache work |
| ⚠️ **Model downloader** | `bigtea-pull <name>` resolves, resumes and reports the fit before downloading. Two models in the catalogue so far |
| ⚠️ **OpenAI-compatible server** | `bigtea-serve` answers `POST /v1/chat/completions`. Localhost only, one request at a time, no streaming yet |
| ⚠️ **Linux and macOS build and pass tests in CI** | but no model has been *run* there yet, and macOS falls back to buffered I/O (`F_NOCACHE` is not wired up) |
| ⚠️ **Prebuilt binaries** | the release workflow is written and asserts the binaries start; not yet fired against a tag |

Architectures implemented: **Qwen3 / Qwen3-MoE** and **DeepSeek-V4-Flash**
(`deepseek4`). Others will load as containers but will not run.

## Building

Bigtea links against a prebuilt **ggml**. It does not vendor it: quantized matmul
kernels are years of specialist SIMD work, and reimplementing them is not where
this project contributes.

### 1. Build ggml

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j
# the static libraries land in build/ggml/src/
```

### 2. Build Bigtea

```bash
git clone https://github.com/aturzone/Bigtea
cd Bigtea

export GGML_LIB_DIR=/path/to/llama.cpp/build/ggml/src   # must contain ggml-base.a, ggml-cpu.a, ggml.a
cargo build --release
cargo test --release
```

`GGML_LIB_DIR` is checked at build time. **Seven of the eight crates build
without it** — the container parser, hardware probe, planner, I/O layer, model
resolver and tokenizer are all useful on a machine that has never compiled a line
of C. Only `bigtea-arch`, the inference engine itself, requires it, and if it is
missing you get one actionable message rather than a wall of unresolved imports:

```
bigtea-arch cannot build: GGML_LIB_DIR is not set.

  1. Build ggml once:
       git clone https://github.com/ggml-org/llama.cpp
       cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
       cmake --build llama.cpp/build --config Release -j

  2. Point Bigtea at the result …
```

<details>
<summary><strong>Windows</strong> — needs the GNU toolchain, not MSVC</summary>

```powershell
rustup default stable-x86_64-pc-windows-gnu
# install MSYS2, then add C:\msys64\mingw64\bin to PATH (needed for libgomp)
$env:GGML_LIB_DIR = "C:/path/to/llama.cpp/build/ggml/src"
cargo build --release
```

If linking fails with `cannot find -lgomp`, MSYS2's `mingw64/bin` is not on PATH.
Note that Git Bash has its own `/mingw64` which is **not** MSYS2's and has no
`gcc` — check with `which gcc`.

**MSYS2 is needed to build, not to run.** On Windows the GNU C++ and OpenMP
runtimes are linked statically, so the resulting `.exe` depends only on system
DLLs and runs on a machine that has never seen MSYS2. Linked dynamically it did
not: Windows killed it with `0xC0000135` (STATUS_DLL_NOT_FOUND) before `main`,
printing nothing at all. Costs ~0.7 MB.

Smart App Control may block freshly built unsigned binaries. `[[bin]]` targets in
this workspace set `test = false` so cargo does not build empty test harnesses
that would be blocked for no reason.
</details>

## Using it

```bash
# What Bigtea can fetch, and what each needs resident
bigtea-pull --list

# Says what it costs and whether it will run here, before downloading
bigtea-pull v4flash --dry-run

# What can this machine run, and what should you close?
bigtea-probe

# Will this model fit, and how fast will it be — before downloading 144 GB
bigtea-model-info model.gguf --budget 8

# Run it
bigtea-run model.gguf "your prompt" -n 32

# Or serve it to a coding agent
bigtea-serve model.gguf --port 8080
```

```console
$ curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json'     -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":6}'
{"id":"bigtea","object":"chat.completion","model":"deepseek-v4-flash",
 "choices":[{"index":0,"message":{"role":"assistant","content":" Paris."},
 "finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}
```

For a split model, pass **any one shard**; the rest are discovered automatically.

| flag | meaning |
|---|---|
| `-n N` | tokens to generate |
| `-b N` | prefill block size (default 2048). Bigger is faster and uses more RAM |
| `-f FILE` | read the prompt from a file, for prompts too long for a command line |
| `--cache GiB` | expert cache budget, overriding the automatic choice |

| environment variable | effect |
|---|---|
| `BIGTEA_THREADS` | threads per graph evaluation (default 12) |
| `BIGTEA_BLOCK_TIMING` | per-block, per-phase timing |
| `BIGTEA_IO_TIMING` | per-tensor read throughput and how much was copied |
| `BIGTEA_SPARSITY` | activation-magnitude histogram |
| `BIGTEA_ROUTING` | how often each expert is selected, and what the hot set would cost |

## How it works

A Mixture-of-Experts container splits cleanly in two. The **always-read** part —
attention, routers, embeddings, shared experts — is touched by every token and is
a small fraction of the file. The **routed experts** are most of the bytes and a
token uses a handful of them.

Bigtea holds the first in RAM and streams the second. For DeepSeek-V4-Flash that
is 7.38 GiB resident against 137 GiB streamed.

Three things make it work rather than thrash:

1. **Weights are bound zero-copy.** `ggml` tensors point at buffers Bigtea already
   owns. A copy would need twice the model and would not fit.
2. **Reads bypass the page cache.** With a model far larger than RAM, a cached
   read costs memory twice — the OS copy and ours — and a "cache hit" under memory
   pressure is a page fault wearing a disguise.
3. **Expert slices are read into their final position.** GGUF pads tensor data to
   32 bytes, not to a disk sector, so aligned buffers can never receive a direct
   transfer. Bigtea deliberately *mis*aligns its destination to match the file's
   sector residue. That took the expert path from 0.80 to 1.58 GiB/s with 0.09% of
   bytes copied instead of 300%.

### Crates

| crate | responsibility |
|---|---|
| `bigtea-gguf` | GGUF container parsing |
| `bigtea-probe` | hardware probing, RAM reclaim advice |
| `bigtea-plan` | fit prediction and residency policy |
| `bigtea-io` | cache-bypassing aligned and skewed reads |
| `bigtea-model` | sharded tensor resolution, partial reads, residency |
| `bigtea-ggml` | FFI to ggml: graphs, zero-copy weight binding |
| `bigtea-tokenizer` | byte-level BPE |
| `bigtea-arch` | architectures and the streaming forward pass |

## Engineering notes

[`docs/graph/`](docs/graph/) holds this project's working notes as a linked graph —
start at [`INDEX.md`](docs/graph/INDEX.md). They are unusually candid on purpose:
several nodes record measurements that **refuted** the idea that motivated them,
and those are kept rather than deleted.

If you read only two:

- [`v4flash-generation-first-numbers.md`](docs/graph/research/v4flash-generation-first-numbers.md)
  — where the current speed goes, and four hypotheses that measurement killed.
- [`zero-copy-expert-reads.md`](docs/graph/research/zero-copy-expert-reads.md)
  — the sector-residue finding, and why the obvious fix was not enough.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: **a performance claim
is not citable until the competing command line and its output are in a document.**
That rule exists because this project broke it once and shipped a false claim for
days.

## Licence

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Bigtea distributes no model weights. Models are yours to obtain, under their own
licences.
