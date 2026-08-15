# The card works — through Vulkan, because CUDA is not installable here

**2026-08-15.** Step 1 of the GPU tier: prove llama.cpp can drive this card
before a line of Rust is written. It can, by **25.6x on prefill**. But not the
way the ticket said, and the reason is worth more than the number.

Links: [gpu-tier-smallest-honest-slice-2026-08-11.md](gpu-tier-smallest-honest-slice-2026-08-11.md) ·
[the-knee-moves-with-n-2026-08-14.md](the-knee-moves-with-n-2026-08-14.md) ·
[v4flash-has-no-slack-2026-08-10.md](v4flash-has-no-slack-2026-08-10.md)

## GPU is still 0%

Nothing below is Bigtea. This is llama.cpp on the same machine in the same
session, which is the *precondition* the ticket set — "if llama.cpp cannot use
the card, we cannot either." It can. The bar does not move until `bigtea-run`
prints a prefill tok/s with the card doing work.

## CUDA is not a toolkit install here, it is a toolchain migration

The ticket said: install CUDA, get `nvcc` answering, rebuild ggml
`-DGGML_CUDA=ON`. That cannot be done on this machine as it stands.

```
$ nvcc --version                                  -> command not found
$ ls "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"   -> does not exist
$ ls "/c/Program Files/Microsoft Visual Studio"             -> does not exist
$ ls "/c/Program Files (x86)/Microsoft Visual Studio"       -> does not exist
```

**`nvcc` on Windows supports only MSVC as its host compiler** — there is no
mingw-w64 support — and there is no MSVC on this machine. Everything the project
builds with is MSYS2 mingw64, including the ggml every test points at:

```
CMAKE_C_COMPILER   = C:/msys64/mingw64/bin/cc.exe
CMAKE_CXX_COMPILER = C:/msys64/mingw64/bin/c++.exe
CMAKE_COMMAND      = C:/msys64/mingw64/bin/cmake.exe
GGML_CUDA:BOOL     = OFF
```

So the CUDA route is: install Visual Studio Build Tools, build `ggml-cuda` with
MSVC, then link an MSVC static library into a **GNU-target** Rust binary —
mixing C++ runtimes, against the `.cargo/config.toml` GNU workaround `CLAUDE.md`
says not to delete. That is a toolchain migration wearing a toolkit install's
clothes, and it belongs in a decision, not in a step 1.

**Vulkan needs none of it.** ggml's Vulkan backend compiles with the compiler
already in use, the driver already ships the loader (`vulkan-1.dll` was present),
and the install was eight MSYS2 packages — vulkan-headers/loader, SPIR-V,
glslang, shaderc. Checked before installing that none of them touch gcc,
binutils or the CRT, because `CLAUDE.md` records that gcc 16.1.0 already needed
a link workaround.

Built into a **separate** directory so the 507 tests keep pointing at the CPU
ggml they always did:

```bash
cmake -S . -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_OPENSSL=OFF -DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00
cmake --build build-vulkan --target llama-completion llama-bench -j 8
```

`-D_WIN32_WINNT=0x0A00` is not optional: vendored `cpp-httplib` calls
`::CreateFile2`, which mingw gates behind Win8, and the build fails there — in
`common`, not in the server, so turning the server off does not avoid it. Both
of the first two build failures were this, and **neither was Vulkan**; the
shader pipeline had generated 98 of 497 shaders before the unrelated stop.

## The card, and the one beside it

```
$ ./build-vulkan/bin/llama-bench.exe --list-devices
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = Intel(R) RaptorLake-S Mobile Graphics Controller | uma: 1 | fp16: 1 | matrix cores: none
ggml_vulkan: 1 = NVIDIA GeForce RTX 3050 6GB Laptop GPU | uma: 0 | fp16: 1 | bf16: 1 | matrix cores: NV_coopmat2
Available devices:
  Vulkan0: Intel(R) RaptorLake-S Mobile Graphics Controller (8045 MiB, 7387 MiB free)
  Vulkan1: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6001 MiB, 5233 MiB free)
```

## The numbers

Qwen3-4B-Q4_K_M (2.32 GiB, fits VRAM with 2.9 GiB spare), llama.cpp build
`daef2b3`, one session, `-r 2`.

```bash
# CPU baseline, the pre-existing CPU-only build
./build/bin/llama-bench.exe -m Qwen3-4B-Q4_K_M.gguf -p 512 -n 128 -t 4,20 -r 2
# GPU
./build-vulkan/bin/llama-bench.exe -m Qwen3-4B-Q4_K_M.gguf --device Vulkan1 -ngl 0,99 -p 512 -n 128 -r 2
./build-vulkan/bin/llama-bench.exe -m Qwen3-4B-Q4_K_M.gguf --device Vulkan0 -ngl 99  -p 512 -n 128 -r 2
```

| config | pp512 (tok/s) | tg128 (tok/s) |
|---|---:|---:|
| CPU, 20 threads | 79.65 ± 5.93 | 3.65 ± 0.10 |
| CPU, 4 threads | 40.25 ± 0.95 | **6.39 ± 0.08** |
| **RTX 3050, `-ngl 99`** | **2042.60 ± 5.52** | **56.53 ± 0.04** |
| Intel iGPU, `-ngl 99` | 38.13 ± 2.09 | 3.26 ± 0.03 |
| RTX 3050, `-ngl 0` | 497.82 ± 243.16 | 3.42 ± 0.08 |

**Against the best CPU configuration of each: prefill 25.6x, generation 8.8x.**

## Two rules, not footnotes

Both of these were live in the table above and both would have inflated the
headline. This project has retracted a competitive claim before, so they are
written as rules.

> **RULE 1 — the baseline must come from the baseline's build.** `-ngl 0` on a
> GPU build is not the CPU path. It is the GPU backend with nothing offloaded,
> and here it reads **3.42 tg128 — *below* the real CPU figure of 6.39** — with
> a ±49% error bar on prefill. Quoting it buys a fake **16x** on generation
> instead of the true 8.8x. A disabled accelerator is not a control.

> **RULE 2 — tune the baseline before you beat it.** The `-t`/`-tb` split is not
> ours alone: on llama.cpp, prefill goes **40.25 → 79.65** from 4 to 20 threads
> and generation goes **6.39 → 3.65** the other way. `llama-bench` defaulted to
> 10 threads on this machine, which is wrong for *both* phases; comparing
> against it would have reported **30.1x instead of 25.6x**. Take the best
> configuration of the thing you are beating, not its default.

Rule 2 carries a second result worth having on its own: **our `-t`/`-tb` finding
reproduces on the reference implementation.** The two-levers-pulling-opposite-ways
shape was measured here on our engine and is visible, at the same crossover, on
llama.cpp — independent confirmation of the threading work rather than a quirk of
our scheduler.

## The iGPU is not a second tier

It has **more free memory than the discrete card** (7387 MiB vs 5233) and it is
UMA, so the upload problem that defines this whole ticket would not exist there.
It is also **slower than the CPU** — 38.13 pp512 against 79.65, 3.26 tg128
against 6.39. No matrix cores, and it shares the DRAM the CPU path is already
saturating.

So the tempting idea — *put the experts on the 8 GB UMA device and skip the
copy* — is dead on arrival, and it cost one command to kill rather than a week.

## What this changes, and what it does not

**Changed:** blocker (a) as recorded in the scoping node — "there is no CUDA
toolkit on this machine at all" — is no longer the gate. It is real for CUDA and
it is now routed around, at the cost of a backend nobody here has written
against.

**Unchanged, and it is still the whole ticket:** weights are bound by writing a
host pointer into `tensor->data` (`weights.rs:286`). A Vulkan tensor lives in a
device buffer filled by `ggml_backend_tensor_set`, which **copies**, exactly as
a CUDA one would. Vulkan removes an MSVC migration from in front of the work; it
does not touch the work.

**Also unchanged:** 76% of a token on the MoE path is disk, and a GPU cannot fix
disk. The 25.6x above is prefill on a model that *fits in VRAM*, which is the
slice the ticket named and the only one this card can plausibly win.

## Open

- **Does `ggml_backend_sched` earn its place before a single-device path
  exists?** The smallest honest Bigtea slice is Qwen3-4B entirely on the device,
  where no mixed-device graph is needed at all.
- **What does the upload cost on a model that fits?** 2.32 GiB at PCIe speed,
  once, at load. Amortised across a session it is probably free; that is a
  measurement, not an assumption.
- **CUDA versus Vulkan on this card, at some later point.** `NV_coopmat2` is
  present, so the Vulkan path is not obviously leaving the tensor cores unused.
  If it were, the fallback is a *dynamically loaded* MSVC-built backend DLL —
  not a static link, and not a toolchain migration.
