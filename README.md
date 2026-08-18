<h1 align="center">Chaos</h1>

<p align="center">
  <img src="assets/logo.png" alt="Chaos" width="180">
</p>

<p align="center">
  <strong>Run models that do not fit in your RAM.</strong><br>
  Keeps the always-read weights resident, streams the routed experts from disk.
</p>

<p align="center">
  <a href="https://github.com/aturzone/Chaos/releases"><img alt="version" src="https://img.shields.io/badge/version-0.0.6-orange"></a>
  <a href="LICENSE"><img alt="licence" src="https://img.shields.io/badge/licence-Apache--2.0-blue"></a>
  <a href="https://github.com/aturzone/Chaos/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/aturzone/Chaos/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="tests" src="https://img.shields.io/badge/tests-679%20passing-brightgreen">
</p>

---

## Start here

**1. Install.**

**Windows** -- download **`Chaos-Setup.exe`** from
[Releases](https://github.com/aturzone/Chaos/releases) and run it. One file with
everything inside it: no archive to unpack, no PowerShell, no administrator
rights. It installs per-user, puts Chaos on your PATH, and adds a Start Menu
entry.

**Linux / macOS** -- unpack the archive and copy the binaries:

```bash
sudo install -m 755 chaos-*/chaos-* /usr/local/bin/ && mkdir -p ~/.chaos/models
```

**2. Get a model.** Ask Chaos for one:

```bash
chaos-pull --list          # 13 models, with the size that actually matters
chaos-pull qwen3-4b        # ~2.5 GB, a good first one
```

...or drop any `.gguf` into `~/.chaos/models` (`%USERPROFILE%\.chaos\models` on
Windows). Nothing is downloaded unless you ask for it.

**3. Run it.**

```bash
chaos-run                                   # lists the models you have
chaos-run qwen3 "The capital of France is"  # any unique part of a name works
```

That is the whole thing. You never type a path, and for a model split across
five shards you never work out which shard to open.

```bash
chaos-run qwen3 "write a haiku" -n 64 --auto   # --auto reads your machine and configures itself
chaos-probe                                    # what can this machine run, and what should you close?
chaos-serve qwen3 --port 8080                  # then open http://127.0.0.1:8080
```

**Or use the window.** `chaos-app` on Windows is a native application: pick a
model, load it, chat, download more. Two colours, no dependencies, a real Win32
window rather than a browser in a frame.

```
INSTALLED  AVAILABLE      v4flash UD-Q4_K_XL  155 GB [5 files]  needs 7.92 GB - streams
                          qwen3-32b Q4_K_M    19.8 GB           needs 19.8 GB - too big
```

**Read the second column, not the first.** A 155 GB model streams on a 16 GB
machine; a 20 GB dense one does not, because a dense container has no routed
experts to leave on disk.

**`chaos-serve` gives you a window as well as a socket.** Open its address in a
browser and you get a chat interface; point a coding agent at
`/v1/chat/completions` and it gets the OpenAI API. Same process, same model
loaded once. The page is compiled into the binary — no CDN, no fonts, no
install, and it works with the network cable out.

<details>
<summary>Two things about your first run, both deliberate</summary>

- **It will be slow once, if it uses the GPU.** ggml compiles its Vulkan shaders
  on first use and the driver caches them afterwards. Measured: 1.63 tok/s on the
  first `--auto` run, **9.0–9.6 tok/s on every run after**. Nothing is wrong.
- **A model whose architecture has never been diffed against llama.cpp is
  refused by name.** That includes `qwen3moe`, so Qwen3-30B-A3B needs `--force`.
  A wrong forward pass produces fluent nonsense rather than an error, so the
  default is refusal.
</details>

---

## What it actually does

Chaos runs a **144 GB** Mixture-of-Experts model on a **15.7 GiB** laptop and
produces correct text, by never loading the model. The weights every token needs
stay in RAM; the routed experts — most of the container, of which a token uses
six of 256 — are read from disk as routing selects them.

```console
$ chaos-run deepseek "The capital of France is" -n 8
model      deepseek4 (direct (cache bypassed))
shape      43 blocks, 4096 embd, 64 heads, 256 experts (6 used, 1 shared)
resident   loaded 534 tensors, 7.24 GiB of 7.24 GiB budget in 4.9s (1.58 GB/s)
           0.14 GiB did not fit and will be re-read from disk on EVERY token (~0.1s each)
           closing these would free up to 1.19 GiB:
             claude.exe                   1.19 GiB (12 processes)
           that is enough to make the whole model resident.
prefill    5 tokens in 8.0s (0.63 tok/s)
output      Paris.
generate   7 tokens in 16.4s (0.428 tok/s, 2.3s per token)
```

That resident block is the point. It is not a progress bar: it is the size of
your shortfall, what it costs per token, and which processes would fix it.

## How fast, honestly

**Every row was measured with both engines alternating in one session**, because
this machine drifts by up to 25% with its own state. Command lines and raw output
for each:
[`where-we-stand-vs-llamacpp-2026-08-16.md`](docs/graph/research/where-we-stand-vs-llamacpp-2026-08-16.md).

| | Chaos | llama.cpp | |
|---|---:|---:|---|
| DeepSeek-V4-Flash (144 GB), prefill ms/token | **1640** | 1679 | parity |
| DeepSeek-V4-Flash, generation | **0.394** | 0.39 | parity |
| Qwen3-30B-A3B, generation | 3.03–3.86 | 3.35 | parity |
| Qwen3-4B, generation, both at defaults | **8.01** | 6.52 | 1.23x ahead |
| Qwen3-4B, generation, both hand-tuned | 7.64 | **9.16** | 1.20x behind |

**Out of the box we lead because we measure your machine and llama.cpp uses a
fixed default. Given equal care on both sides llama.cpp is faster on dense
models.** On models that stream from disk, the two are level.

If you want the fastest local inference today, use
[llama.cpp](https://github.com/ggml-org/llama.cpp). Chaos is worth your time if
you want an engine that owns residency and tells you the truth about your
machine.

<details>
<summary>Three claims that are retracted and must not be repeated</summary>

- ~~"Chaos runs models larger than RAM and llama.cpp cannot."~~ It can, with
  `--no-repack`. Larger-than-RAM is not the differentiator.
- ~~"Generation is ~2x behind on Qwen3-30B."~~ Re-measured alternating: parity.
- ~~"On V4-Flash: prefill 1.62x behind, generation 3-4x behind."~~ That stood in
  this README for nine days. Re-measured: parity on both. It is a correction in
  our favour, which is not better — and the discarded warm-up in that session
  read llama.cpp at 0.23 tok/s, which would have made it a 1.7x *lead*. It is
  not one.
</details>

## Progress

Every bar is a ratio of two counted things, both named. Filled cells are floored,
never rounded up.

```
CLI flags         91%  [##################..]  165 of llama.cpp's 182
Chat templates    96%  [###################.]  52 of its 54 names
Tokenizers        83%  [################....]  5 of 6 families
Samplers          80%  [################....]  16 of 20
Architectures      9%  [#...................]  13 of the 141 it declares
GPU backends      20%  [####................]  1 of 5, Vulkan only
Browser UI        33%  [######..............]  2 of 6 things a chat UI needs
V4-Flash speed     2%  [....................]  0.394 of 20 tok/s
```

Where those numbers come from, in the same order: 17 flags are declined with a
written reason and **0 are unrecognised**; the two missing templates are Hunyuan
variants; the tokenizer families are BPE, SPM, WordPiece, Unigram and RWKV; the
sampler audit was 2026-08-11; **every one of the 13 architectures was diffed
against llama.cpp at 8 prompts**, which is what counts as verified here; the
Vulkan device path is bound but **not** verified — it fails 1 of those 8 prompts
where the CPU path fails none; and the UI **sends and streams**, with no model
picker, stop button, saved history or sampler controls.

The first four are enforced by tests rather than counted by hand
(`refused_flags_decline`, `jinja_agrees_with_families`), which is why they have
stopped drifting.

The last bar is the honest one. **`V4-Flash speed` will not move, and we
measured how far it cannot move.** A token is 1.56 s of expert reading plus
**0.84 s that never touches the disk** — so with the entire 144 GB model resident
in RAM, this CPU tops out at **1.19 tok/s**. 20 tok/s is a 50 ms token: the fixed
cost alone is 17x over budget, and it separately needs 67.7 GB/s of bandwidth to
the expert weights. That is a GPU-memory specification, not a code change.

So here is the thing nobody had published — **tok/s against resident RAM for a
144 GB model**, which Chaos can sweep because it owns residency where an `mmap`
engine cannot be told to use exactly N GiB:

```
    RAM     experts resident    tok/s
   16 GB           ~3%          0.42   <- measured on the laptop above
   64 GB          ~38%          0.55
  128 GB          ~85%          0.93
  160 GB          100%          1.19   <- the ceiling, nothing left to stream
```

**Holding the whole model in RAM is worth 2.9x, not 48x.** That is a useful
answer even though it is not the hoped-for one, and it is the kind of thing this
project exists to tell you before you spend money.
[`v4flash-ram-frontier-2026-08-16.md`](docs/graph/research/v4flash-ram-frontier-2026-08-16.md)
has the sweep, the fit (R² = 0.997) and the controls.

## Build from source

```bash
# ggml once -- Chaos borrows its kernels rather than rewriting years of SIMD work
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build llama.cpp/build --config Release -j

git clone https://github.com/aturzone/Chaos && cd Chaos
export GGML_LIB_DIR=$PWD/../llama.cpp/build/ggml/src
cargo build --release
```

Clean clone to working binaries: **23 s**. Nine of the ten crates build without
`GGML_LIB_DIR` at all, and CI proves it.

<details>
<summary><strong>Windows</strong> — GNU toolchain, not MSVC</summary>

```powershell
rustup default stable-x86_64-pc-windows-gnu
# install MSYS2, then put C:\msys64\mingw64\bin on PATH (for libgomp)
$env:GGML_LIB_DIR = "C:/path/to/llama.cpp/build/ggml/src"
cargo build --release
```

`cannot find -lgomp` means MSYS2's `mingw64/bin` is not on PATH. Git Bash has its
own `/mingw64` which is **not** MSYS2's and has no `gcc` — check `which gcc`.

**MSYS2 is needed to build, not to run**: the GNU C++ and OpenMP runtimes link
statically, so the `.exe` runs on a machine that has never seen it. Linked
dynamically it died with `0xC0000135` before `main`, printing nothing.

`.cargo/config.toml` sets `link-self-contained=no` and must stay — MSYS2 gcc
16.1.0 dropped symbols rustup's bundled `crt2.o` still references.
</details>

<details>
<summary><strong>GPU</strong> — a second ggml build</summary>

The CPU build has no Vulkan archive, and the GPU tests **skip** rather than fail
without a card, so a green run proves nothing about the device path. Build ggml
again with `-DGGML_VULKAN=ON` into its own directory and point `GGML_LIB_DIR`
there for any work touching `--device`, `-ngl` or `-ot`.
</details>

## Going deeper

| | |
|---|---|
| [`STATUS.md`](STATUS.md) | the scoreboard, what works, what is left. Read before quoting any number |
| [`docs/graph/INDEX.md`](docs/graph/INDEX.md) | every working note, one line each. Many record measurements that **refuted** the idea that motivated them |
| [`CHANGELOG.md`](CHANGELOG.md) | including the retractions |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **a performance claim is not citable until the competing command line and its output are in a document** |

If you read only three notes:
[`v4flash-has-no-slack`](docs/graph/research/v4flash-has-no-slack-2026-08-10.md)
(four attempts to find redundancy in a 144 GB model, four negatives),
[`the-plateau-was-ours`](docs/graph/research/the-plateau-was-ours-2026-08-10.md)
(two "facts" that were ceilings we had built), and
[`threads-were-never-plumbed`](docs/graph/research/threads-were-never-plumbed-2026-08-10.md)
(a sweep whose knob was disconnected, and therefore indistinguishable from a flat
response).

## Licence

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Chaos distributes no
model weights. Models are yours to obtain, under their own licences.
