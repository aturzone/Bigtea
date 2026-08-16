# Prompt for the next session

Copy everything below the line into a fresh session in `C:\Projects\Chaos`.

---

You are continuing work on **Chaos**, a Rust inference runner for GGUF models
larger than RAM. Read `CLAUDE.md` and `STATUS.md` first — `STATUS.md` is the
canonical scoreboard. Then read only the 2–3 graph nodes a task links to.

Today's job is **the release**: finish the CLI, rebrand the whole project, fix
every document, and cut a tested Windows build. Work on `ticket/<name>` branches
with PRs, merge when CI is green, keep `main` verified.

## Where things stand

`main` is green: **561 tests, 0 failed**, workspace clippy clean.
**167 implemented CLI flags, 17 declined** — every declined one carries a written
reason, and that table was audited three times in one day after six reasons had
rotted. Audit it again whenever you change what the engine can do.

Build (note the **two different** ggml directories):

```
export GGML_LIB_DIR=C:/Projects/llamacpp-unsloth/build/ggml/src          # CPU only
export GGML_LIB_DIR=C:/Projects/llamacpp-unsloth/build-vulkan/ggml/src   # GPU work
export PATH="/c/msys64/mingw64/bin:$PATH"
cargo test --release
```

**The CPU build has no Vulkan archive and the GPU tests SKIP rather than fail
without a card.** A green "6 passed" was once reported for a file whose two GPU
tests had never run. Use `build-vulkan` for anything touching the device.

## The measurements you must not contradict

All taken 2026-08-16 on an RTX 3050 6 GB / i7-13650HX / 15.7 GiB machine.

| | Chaos | llama.cpp | notes |
|---|---|---|---|
| Qwen3-30B generation | 3.03–3.86 | 3.35 | **parity**, paired 3–2, ranges overlap |
| Qwen3-30B prefill | 1.22 | 1.17 | parity |
| Qwen3-4B `-ngl` frontier | 43.3 → 77.3 prefill | — | 1.79x, monotonic, no knee |

- `CLAUDE.md`'s old "generation is ~2x behind (1.07 vs 2.16)" is **retracted**.
  Do not quote it. Do not claim a lead either.
- **The GPU is a dense-model dial.** On streaming MoE it is **4.3x slower**
  (2.61 → 0.61 tok/s) because experts run on the host whatever `-ngl` says.
- `--op-offload` works and is **19% slower** — we submit ~180 graphs per pass, so
  weight copies amortise over a block, not a model.
- Parallel experts gave **1.29x on expert compute, 1.10x end to end**.
- Overlapping expert reads with compute is **1.03x** — built, measured, reverted.
- Expert cache plateaus at 6 GiB (2/4/6/8 → 2.22/2.66/3.45/3.43 tok/s).

**This machine now drifts by more than most effects.** Anything worth under ~10%
needs both arms alternating in one session, or it cannot be assessed. Comparing
against a number from an earlier session is how three wrong figures got published
here.

## The checklist

### 1. Rebrand the project to **chaos**

Complete and mechanical: crate names, binary names (`chaos-run` → `chaos-run`
etc.), every string in `--help` and the info lines, all 104 `.md` files, the
release workflow, `Cargo.toml` metadata. Leave the git remote alone — Atur
renames the repo himself afterwards.

**The logo:** Atur has an SVG. **Ask him for its path before starting** — there
is no `.svg` anywhere in the repo and it was not attached. Give it a **white
background**. Render it in the terminal when the CLI starts: Unicode half-blocks
(`▀` with foreground/background colours) give two vertical pixels per cell and
is the closest to pixel-perfect a terminal allows. Rasterise the SVG **at build
time into a small embedded bitmap** — do not add an SVG parser or a runtime
dependency to a workspace that currently has **zero external dependencies**.
Honour `NO_COLOR` and a non-TTY stdout by skipping the banner.

### 2. Every `.md` file, no misses

104 files. Names, commands, paths, and any number that today's measurements
changed. `docs/graph/INDEX.md` is read first by every session, so its entries
must match the nodes they point at.

### 3. `README.md`

Simple install instructions, and progress bars at the end. **No emoji in any
bar** — Atur was explicit. Use ASCII blocks, e.g.:

```
CLI            [##################  ]  92%
Architectures  [##                  ]   9%   13 of 141 verified
BIG BANG       [#####               ]  25%
GUI            [                    ]   0%   not planned, CLI only
```

Bars must be defensible from `STATUS.md`, not decorative. "GUI 0%" is correct
and should say "not planned".

### 4. An honest status against llama.cpp

A real table showing where we win, where we lose, and where it is parity — using
the numbers above. It is fine to write that we intend to close the gaps; it is
not fine to imply a lead we do not have. Every competitive claim needs the
opposing command line and its output in a doc.

### 5. Test from zero

Clone-to-run on a clean checkout: build, run a small model, run a heavy one,
`--auto`, `chaos-serve`. Write down what a fresh machine actually needs.

### 6. Release

`.github/workflows/release.yml` already builds Windows/Linux/macOS and packages
them. Windows needs the **GNU** toolchain plus MSYS2 mingw64. Add:

- a Windows install script that puts binaries on `PATH` and creates a **models
  directory** — Atur will drop `.gguf` files there by hand, no internet
- **in-place update**: re-running the installer over an older version must
  upgrade cleanly

**No GUI. No `.exe` installer with a UI.** CLI only.

## What NOT to promise

**V4-Flash (144.4 GiB) will not run at coding-agent speed, on either machine.**
`v4flash-has-no-slack-2026-08-10.md` closed this with arithmetic: 20 tok/s needs
79 MB/token, V4-Flash reads 3288, everything still alive multiplies to 3.1x
against a 42x gap. On the 64 GiB machine it still streams — 144 GiB does not fit
— and 32 GiB of VRAM against 144 GiB of weights is the same wall.

Atur wants "impossible things", and the honest impossible-thing here is
different: **Qwen3-30B-A3B at 17.3 GiB fits entirely inside a 5090's 32 GiB
VRAM.** That is the demo. Build BIG BANG around that.

If asked to make V4-Flash fast, say plainly that the measurement closed it and
offer the 30B instead.

## Models already on disk

```
C:\Projects\models\qwen3moe\Qwen3-30B-A3B-Q4_K_M.gguf      17.3 GiB   the demo
C:\Projects\models\qwen3-4b\Qwen3-4B-Q4_K_M.gguf            2.3 GiB   sanity check
C:\Projects\models\llama32-1b\Llama-3.2-1B-Instruct-...     0.7 GiB   parity harness
C:\Projects\models\v4flash\DeepSeek-V4-Flash-...00005       144.4 GiB  larger-than-RAM demo only
```

## Rules that were each bought with a mistake

- **Never `git push -u`** — it writes the token into `.git/config`. Push with the
  token from `C:\Projects\.env` inline in the URL, output redacted, never echoed.
- Verify containment with `git merge-base --is-ancestor <branch> origin/main`
  *before* deleting a branch.
- After merging, fast-forward `main` explicitly and **re-run tests on `main`
  itself**, then check a file only the merge added.
- GitHub parses only the *first* issue in `Closes #1, #2, #3`.
- **ggml aborts rather than returning errors** — an exhausted arena, an
  out-of-range device index, a misaligned host pointer. Check on the Rust side.
- **A wrong forward pass produces fluent nonsense, never a crash.** Loading is
  not evidence; answering in English is not evidence. Only a diff against
  llama.cpp counts.
- Correct Atur's English briefly at the top of every reply.
