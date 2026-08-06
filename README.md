<h1 align="center">Bigtea</h1>

<p align="center">
  <strong>A CPU inference runner for models that do not fit in RAM.</strong><br>
  Keeps the always-read weights resident, streams routed experts from disk per token.
</p>

<p align="center">
  <a href="#status"><img alt="status" src="https://img.shields.io/badge/status-v0.0.0%20preview-orange"></a>
  <a href="LICENSE"><img alt="licence" src="https://img.shields.io/badge/licence-Apache--2.0-blue"></a>
  <a href="#building"><img alt="rust" src="https://img.shields.io/badge/rust-1.82%2B-informational"></a>
  <img alt="tests" src="https://img.shields.io/badge/tests-157%20passing-brightgreen">
</p>

---

Bigtea runs a **144 GB** Mixture-of-Experts model on a **15.7 GiB** laptop, on CPU,
and produces correct text. It does that by never loading the model: the weights
every token needs stay in RAM, and the routed experts — which are most of the
container and of which a token uses six of 256 — are read from disk as routing
selects them.

```console
$ bigtea-run DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf "The capital of France is" -n 5
model      deepseek4 (direct (cache bypassed))
shape      43 blocks, 4096 embd, 64 heads, 256 experts (6 used, 1 shared)
resident   loaded 101 tensors, 3.97 GiB of 3.97 GiB budget in 3.4s (1.24 GB/s)
           3.40 GiB will be re-read from disk on EVERY token (~3.0s each)
           closing these would free up to 5.63 GiB:
             Code.exe                     1.55 GiB (15 processes)
             Telegram.exe                 0.66 GiB
           that is enough to make the whole model resident.
prefill    5 tokens in 20.9s (0.24 tok/s)
output      Paris.
generate   4 tokens in 95.6s (0.042 tok/s, 23.9s per token)
```

---

## Read this before you try it

**Bigtea is currently slower than llama.cpp at generation.** On the model above we
measure **0.042 tok/s against llama.cpp's 0.45** — about ten times slower. On
Qwen3-30B-A3B generation we are roughly **2x behind** (1.07 vs 2.16 tok/s).

**llama.cpp can also run models larger than RAM**, given `--no-repack`. "Runs a
model bigger than your memory" is not a thing only Bigtea does, and this project
spent days believing otherwise because nobody ran the opposing command. That
claim is retracted, in writing, in
[`docs/graph/research/head-to-head-llamacpp-2026-08-05.md`](docs/graph/research/head-to-head-llamacpp-2026-08-05.md).

Where Bigtea is currently **ahead** is prefill on Qwen3-30B-A3B:

| prompt tokens | Bigtea | llama.cpp | |
|---:|---:|---:|:--|
| 565 | **27.64** | 23.55 | ahead |
| 2206 | **36.60** | 33.59 | ahead |
| 4395 | 38.40 | 40.25 | 95% — behind |
| 8775 | 34.88 | 35.01 | 99.6% — parity |
| 4395 (`-b 4096`) | **43.61** | 40.25 | ahead |

Both engines produce identical, correct output at every length; llama.cpp is
measured with a fully warm page cache.

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

**v0.0.0 — preview.** The engine works and is verified. The product around it is
not built yet.

| | |
|---|---|
| ✅ Runs models several times larger than RAM, on CPU | Qwen3-30B-A3B (17.28 GiB) and DeepSeek-V4-Flash (144 GB) both on a 15.7 GiB machine |
| ✅ Correct output, verified against llama.cpp | 157 unit tests + 16 container-backed tests |
| ✅ Cache-bypassing direct I/O with zero-copy expert reads | |
| ✅ Honest reporting of residency, throughput and shortfalls | |
| ⚠️ **Generation is slow** | see above — no KV cache yet on the V4-Flash path |
| ❌ No model downloader | you bring your own `.gguf` |
| ❌ No server / API | no OpenAI-compatible endpoint yet |
| ❌ **Linux and macOS are untested** | the code has paths for both; only Windows has been run. CI covers the build |
| ❌ No prebuilt binaries | you build from source, and you need ggml |

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
cmake -B build -DCMAKE_BUILD_TYPE=Release
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
       cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
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

Smart App Control may block freshly built unsigned binaries. `[[bin]]` targets in
this workspace set `test = false` so cargo does not build empty test harnesses
that would be blocked for no reason.
</details>

## Using it

```bash
# What can this machine run, and what should you close?
bigtea-probe --quick

# Will this model fit, and how fast will it be — before downloading 144 GB
bigtea-model-info model.gguf --budget 8

# Run it
bigtea-run model.gguf "your prompt" -n 32
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
