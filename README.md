<h1 align="center">Chaos</h1>

<p align="center">
  <strong>An inference runner for models that do not fit in RAM.</strong><br>
  Keeps the always-read weights resident, streams routed experts from disk per token.
</p>

<p align="center">
  <a href="#where-it-stands-against-llamacpp"><img alt="status" src="https://img.shields.io/badge/status-v0.0.2%20preview-orange"></a>
  <a href="LICENSE"><img alt="licence" src="https://img.shields.io/badge/licence-Apache--2.0-blue"></a>
  <a href="#build-it-yourself"><img alt="rust" src="https://img.shields.io/badge/rust-1.82%2B-informational"></a>
  <a href="https://github.com/aturzone/Chaos/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/aturzone/Chaos/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="tests" src="https://img.shields.io/badge/tests-566%20passing-brightgreen">
</p>

---

> **[STATUS.md](STATUS.md)** is the current state of the project in one page — the
> scoreboard against llama.cpp, what works, the known limits, and what is next.
> Read it before quoting any number from here.

Chaos runs a **144 GB** Mixture-of-Experts model on a **15.7 GiB** laptop and
produces correct text. It does that by never loading the model: the weights every
token needs stay in RAM, and the routed experts — most of the container, of which
a token uses six of 256 — are read from disk as routing selects them.

```console
$ chaos-run DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf "The capital of France is" -n 8
model      deepseek4 (direct (cache bypassed))
shape      43 blocks, 4096 embd, 64 heads, 256 experts (6 used, 1 shared)
resident   loaded 286 tensors, 6.53 GiB of 6.53 GiB budget in 4.9s (1.44 GB/s)
           0.85 GiB did not fit and will be re-read from disk on EVERY token (~0.6s each)
           closing these would free up to 1.94 GiB:
             chrome.exe                   1.40 GiB (13 processes)
             Telegram.exe                 0.54 GiB
           that is enough to make the whole model resident.
prefill    5 tokens in 7.9s (0.63 tok/s)
output      Paris.
generate   8 tokens in 20.2s (0.396 tok/s, 2.5s per token)
```

That last block is the point of the project. It is not a progress bar: it is the
size of your shortfall, what the shortfall costs per token, and the names of the
processes that would fix it.

---

## Install

### Prebuilt binaries

Download the archive for your platform from
[Releases](https://github.com/aturzone/Chaos/releases), unpack it, and put the
binaries somewhere on `PATH`. They carry no runtime dependencies beyond the
system libraries — on Windows the GNU C++ and OpenMP runtimes are linked
statically, so the `.exe` runs on a machine that has never seen MSYS2.

**Windows**, from the unpacked folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

That copies the binaries to `%LOCALAPPDATA%\Chaos\bin`, adds it to your user
`PATH`, and creates `%USERPROFILE%\.chaos\models` for your `.gguf` files. Run it
again over a newer version and it upgrades in place. `-Prefix` and `-ModelsDir`
override either location; `-Uninstall` reverses it.

**Linux and macOS**:

```bash
tar xzf chaos-*-linux-x86_64.tar.gz
sudo install -m 755 chaos-*/chaos-* /usr/local/bin/
mkdir -p ~/.chaos/models
```

Then drop `.gguf` files into the models directory — Chaos never downloads
anything you did not ask it to, and `chaos-run` takes a path, so the directory is
a convention rather than a requirement.

### Build it yourself

Chaos links against a prebuilt **ggml**. It does not vendor it: quantized matmul
kernels are years of specialist SIMD work, and rewriting them is not where this
project contributes.

```bash
# 1. build ggml once
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build llama.cpp/build --config Release -j
#    the static libraries land in llama.cpp/build/ggml/src/

# 2. build Chaos
git clone https://github.com/aturzone/Chaos
cd Chaos
export GGML_LIB_DIR=$PWD/../llama.cpp/build/ggml/src   # needs ggml-base.a, ggml-cpu.a, ggml.a
cargo build --release
cargo test --release
```

`GGML_LIB_DIR` is checked at build time. **Nine of the ten crates build without
it** — the container parser, hardware probe, planner, I/O layer, model resolver,
tokenizer, grammar and template engine are all useful on a machine that has never
compiled a line of C. Only `chaos-arch`, the inference engine itself, needs it,
and when it is missing you get one actionable message rather than a wall of
unresolved imports.

<details>
<summary><strong>Windows</strong> — needs the GNU toolchain, not MSVC</summary>

```powershell
rustup default stable-x86_64-pc-windows-gnu
# install MSYS2, then add C:\msys64\mingw64\bin to PATH (needed for libgomp)
$env:GGML_LIB_DIR = "C:/path/to/llama.cpp/build/ggml/src"
cargo build --release
```

If linking fails with `cannot find -lgomp`, MSYS2's `mingw64/bin` is not on PATH.
Git Bash has its own `/mingw64` which is **not** MSYS2's and has no `gcc` — check
with `which gcc`.

**MSYS2 is needed to build, not to run.** Linked dynamically, the binary died
with `0xC0000135` before `main` on any machine without MSYS2, printing nothing at
all. Static linking costs ~0.7 MB and fixes it.

`.cargo/config.toml` sets `link-self-contained=no` and must stay: MSYS2 gcc
16.1.0 dropped symbols rustup's bundled `crt2.o` still references, so without it
every link fails with "undefined reference" on code that compiles.

Smart App Control may block freshly built unsigned binaries. `[[bin]]` targets in
this workspace set `test = false` so cargo does not build empty test harnesses
that would be blocked for no reason.
</details>

<details>
<summary><strong>GPU</strong> — a second ggml build</summary>

The CPU build above has no Vulkan archive, and the GPU tests **skip** rather than
fail without a card, so a green run proves nothing about the device path. Build
ggml again with `-DGGML_VULKAN=ON` into a separate directory and point
`GGML_LIB_DIR` at that one for any work touching `--device`, `-ngl` or `-ot`.
</details>

## Using it

```bash
chaos-probe                        # what can this machine run, and what should you close?
chaos-model-info model.gguf --budget 8   # will it fit, and how fast — before downloading 144 GB
chaos-run model.gguf "your prompt" -n 32
chaos-run model.gguf "your prompt" --auto # pick the device, -ngl and the cache from this machine
chaos-serve model.gguf --port 8080        # OpenAI-compatible server
```

For a split model, pass **any one shard**; the rest are discovered automatically.

Two things about your first run, both deliberate:

- **It will be slow, once, if it uses the GPU.** ggml's Vulkan backend compiles
  its shader set on first use and the driver then caches it on disk. Measured:
  1.63 tok/s on the first `--auto` run of a fresh install, **9.0–9.6 tok/s on
  every run after it**. Nothing is wrong.
- **A model whose architecture has not been diffed against llama.cpp is
  refused**, by name, with the list of the 13 that have. That includes
  `qwen3moe`, so **Qwen3-30B-A3B needs `--force`** — it scores 2 exact, 4
  near-tie and 2 outside the band on the eight-prompt sweep, and this runner will
  not pretend that is verified. A wrong forward pass produces fluent nonsense,
  never an error, which is why the default is refusal.

```console
$ curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":6}'
{"id":"chaos","object":"chat.completion","model":"Llama-3.2-1B-Instruct",
 "choices":[{"index":0,"message":{"role":"assistant","content":"Paris."},
 "finish_reason":"stop"}],"usage":{"prompt_tokens":41,"completion_tokens":3,"total_tokens":44}}
```

`chaos-run --help` lists all 165 flags. A handful worth knowing:

| flag | meaning |
|---|---|
| `-n N` | tokens to generate |
| `-b N` | prefill block size. Bigger is faster and uses more RAM |
| `-f FILE` | read the prompt from a file |
| `-t N` / `-tb N` | threads for generation / for prefill. **They want opposite counts** |
| `--cache GiB` | expert cache budget, overriding the automatic choice |
| `-ngl N` | layers to place on the GPU. A dial, not a switch — see below |
| `--auto` | read the machine and choose device, `-ngl` and cache without being told |

| environment variable | effect |
|---|---|
| `CHAOS_BLOCK_TIMING` | per-block, per-phase timing |
| `CHAOS_IO_TIMING` | per-tensor read throughput and how much was copied |
| `CHAOS_ROUTING` | how often each expert is selected, and what a hot set would cost |
| `CHAOS_NO_BANNER` | skip the startup logo (as does `NO_COLOR`, and any non-terminal output) |

## Where it stands against llama.cpp

**Every row below was measured with both engines alternating in one session**,
because this machine drifts by up to 25% with its own state and comparing against
a number from an earlier session is how three wrong figures got published here.
Command lines and raw output for each row are in
[`where-we-stand-vs-llamacpp-2026-08-16.md`](docs/graph/research/where-we-stand-vs-llamacpp-2026-08-16.md).

All figures: i7-13650HX, 15.7 GiB RAM, RTX 3050 6 GB, NVMe, 2026-08-16.

| workload | Chaos | llama.cpp | verdict |
|---|---:|---:|---|
| **DeepSeek-V4-Flash (144 GB)**, prefill, ms/prompt token | **1640** | 1679 | **parity** |
| **DeepSeek-V4-Flash**, generation | **0.394** | 0.39 | **parity** |
| Qwen3-30B-A3B, generation (streams from disk) | 3.03–3.86 | 3.35 | **parity** — paired 3–2, ranges overlap |
| Qwen3-30B-A3B, prefill | 1.22 | 1.17 | **parity** |
| Qwen3-4B, generation, both at their defaults | **8.01** | 6.52 ± 0.33 | **1.23x ahead** |
| Qwen3-4B, generation, both hand-tuned | 7.64 | **9.16 ± 0.43** | **1.20x behind** |
| Qwen3-4B, prefill | 83.4 | **88.3** | 1.06x behind |
| Llama-3.2-1B, generation, both hand-tuned | 21.95 | **27.85 ± 1.98** | **1.27x behind** |

**Where we are ahead is out of the box, and the reason is not a faster kernel.**
Chaos measures the machine and picks its thread counts; llama.cpp uses a fixed
default. Given equal care on both sides llama.cpp is still faster on the dense
path, and that ratio (1.20x) matches what was recorded before any of the thread
work, which is what says it is real rather than an artefact of where on the curve
each engine was sitting.

**On the streaming MoE model the two are level**, and neither side should claim
otherwise: the ranges overlap almost completely and both decline across a long
session as the machine warms.

### Coverage — this is the real gap, and it is not close

| | Chaos | llama.cpp | |
|---|---:|---:|---|
| CLI flags, long form | **165** implemented, 17 declined with a written reason | 182 | 0 unrecognised |
| chat template families | **52** | 54 | missing `hunyuan-dense`, `hunyuan-vl` |
| tokenizer families | **5** — BPE, SPM, WordPiece, Unigram, RWKV | 6 | missing `plamo2` |
| architectures **diffed against the reference** | **13** | 141 declared | the big one |
| samplers | 16 | 20 | audited 2026-08-11 |
| GPU backends | **1**, Vulkan, **not verified** | CUDA, Metal, Vulkan, SYCL, HIP | |

The architecture number is not comparable as written: llama.cpp *declares* 141,
and Chaos's 13 are the ones whose output was diffed token for token against it at
eight prompts each. Nobody has checked all 141. But 13 is still 13.

**The honest one-line answer: on this machine, for the models we support, Chaos
is as fast as llama.cpp and supports far less of the world.** If you want the
fastest local inference today, use
[llama.cpp](https://github.com/ggml-org/llama.cpp).

### Three claims that are retracted and must not be repeated

- ~~"Chaos runs models larger than RAM and llama.cpp cannot."~~ It can, given
  `--no-repack`. Larger-than-RAM is not the differentiator; **tok/s at a stated
  footprint under an owned residency policy** is.
- ~~"Generation is ~2x behind on Qwen3-30B (1.07 vs 2.16)."~~ Re-measured
  2026-08-16 with both engines alternating: **parity**. The 2x figure is dead, and
  it had been steering which work got picked.
- ~~"Chaos leads llama.cpp on V4-Flash load and prefill."~~ Chaos's numbers were
  fresh and llama.cpp's were copied from a two-day-old document.
- ~~"On V4-Flash: prefill 1.62x behind, generation 3-4x behind."~~ That stood in
  this README for nine days. Re-measured 2026-08-16, alternating: **parity on
  both**. It is a correction in our favour, which is not better — the same rule
  caught it. The discarded warm-up in that session read llama.cpp at 0.23 tok/s,
  which would have made it a 1.7x *lead*; it is not one.

### The GPU is a dense-model dial, not a speedup

`-ngl` on Qwen3-4B is smooth and monotonic with no knee — **1.79x on prefill,
1.40x on generation** from 0 to 99 layers. On Qwen3-30B-A3B, the model this
project exists for, the same flag is **4.3x slower** (2.61 → 0.61 tok/s), because
76% of a token is disk and the experts run on the host whatever `-ngl` says.
`chaos-run` warns, with the measurement in the message, when a device is opened
on a model that streams experts.

**A speedup measured on a model that fits does not transfer to one that does
not.** Every GPU number this project has published was measured on a model that
fits, and none of them predicted that one.

## How it works

A Mixture-of-Experts container splits cleanly in two. The **always-read** part —
attention, routers, embeddings, shared experts — is touched by every token and is
a small fraction of the file. The **routed experts** are most of the bytes and a
token uses a handful.

Chaos holds the first in RAM and streams the second. For DeepSeek-V4-Flash that
is 7.38 GiB resident against 137 GiB streamed.

Four things make that work rather than thrash:

1. **Weights are bound zero-copy.** `ggml` tensors point at buffers Chaos already
   owns. A copy would need twice the model and would not fit.
2. **Reads bypass the page cache.** With a model far larger than RAM a cached read
   costs memory twice, and a "cache hit" under memory pressure is a page fault
   wearing a disguise.
3. **Expert slices are read into their final position.** GGUF pads tensor data to
   32 bytes, not to a disk sector, so a conventionally aligned buffer can never
   receive a direct transfer. Chaos deliberately *mis*aligns its destination to
   match the file's sector residue: 0.80 → 1.58 GiB/s, with 0.09% of bytes copied
   instead of 300%.
4. **Every reader gets its own file handle.** A Windows handle without
   `FILE_FLAG_OVERLAPPED` is synchronous, so N threads on one handle hold the
   drive at queue depth 1. Eight pooled handles: 2.01 → 2.69 GiB/s.

The cache admits by **frequency, never recency**. Expert access is a cyclic scan,
so layer 0's slices are always the oldest thing present when layer 47 needs room —
recency evicts them immediately before they are wanted again. Frequency-gated
admission took the hit rate from 17% to 70% at the same budget.

### Crates

| crate | responsibility |
|---|---|
| `chaos-gguf` | GGUF container parsing |
| `chaos-probe` | hardware probing, RAM reclaim advice |
| `chaos-plan` | fit prediction and residency policy |
| `chaos-io` | cache-bypassing aligned and skewed reads |
| `chaos-model` | sharded tensor resolution, partial reads, residency |
| `chaos-ggml` | FFI to ggml: graphs, scheduler, zero-copy weight binding |
| `chaos-tokenizer` | five tokenizer families and the chat templates |
| `chaos-grammar` | GBNF and JSON-schema constraints |
| `chaos-jinja` | chat template rendering |
| `chaos-arch` | architectures and the streaming forward pass |

**Zero external dependencies**, deliberately, in all ten. The startup logo is
rasterised offline by `tools/rasterise-logo.py` into 3 KB of committed bytes
rather than parsed at runtime, for the same reason.

## Engineering notes

[`docs/graph/`](docs/graph/) holds this project's working notes as a linked
graph — start at [`INDEX.md`](docs/graph/INDEX.md). They are unusually candid on
purpose: many nodes record measurements that **refuted** the idea that motivated
them, and those are kept rather than deleted.

If you read only three:

- [`v4flash-has-no-slack-2026-08-10.md`](docs/graph/research/v4flash-has-no-slack-2026-08-10.md)
  — four independent attempts to find redundancy in a 144 GB model, four
  negatives, and why 20 tok/s is not a code problem.
- [`the-plateau-was-ours-2026-08-10.md`](docs/graph/research/the-plateau-was-ours-2026-08-10.md)
  — two written-down "facts" that turned out to be ceilings we had built.
- [`threads-were-never-plumbed-2026-08-10.md`](docs/graph/research/threads-were-never-plumbed-2026-08-10.md)
  — a sweep whose knob was disconnected, and therefore indistinguishable from a
  flat response.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: **a performance claim
is not citable until the competing command line and its output are in a
document.** That rule exists because this project broke it once and shipped a
false claim for days.

## Licence

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Chaos distributes no model weights. Models are yours to obtain, under their own
licences.

---

## Progress

Every bar is a ratio of two counted things, both stated. Nothing here is a
feeling about how far along something is.

```
CLI flags       [##################  ]  91%   165 of llama.cpp's 182 implemented, 17 declined
                                              with a written reason, 0 unrecognised
Chat templates  [################### ]  96%   52 of its 54 built-in names
Tokenizers      [#################   ]  83%   5 of 6 families: BPE, SPM, WordPiece, Unigram, RWKV
Samplers        [################    ]  80%   16 of 20, audited 2026-08-11
Architectures   [##                  ]   9%   13 of the 141 llama.cpp declares, each diffed
                                              against it at 8 prompts
GPU backends    [####                ]  20%   1 of 5, Vulkan, and the device path is NOT verified
V4-Flash speed  [                    ]   2%   0.396 of the 20 tok/s target. Closed by measurement,
                                              not by effort: 20 tok/s needs 79 MB/token and this
                                              model reads 3288
GUI             [                    ]   0%   not planned. CLI only
```

The last two are the honest ones. **`V4-Flash speed` will not move**, and that is
a finding rather than a backlog item: everything still alive multiplies to 3.1x
against a 42x gap, and the remaining cost is the active weights coming from disk,
which no amount of code changes. The number that *can* move is the one nobody has
published — **tok/s against resident bytes for a 144 GB model** — and Chaos can
sweep it because it owns residency, where an `mmap` engine cannot be told to use
exactly N GiB.
