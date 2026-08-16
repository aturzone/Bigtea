# Prompt for the next session

Copy everything below the line into a fresh session in the repository root.

---

You are continuing work on **Chaos**, a Rust inference runner for GGUF models
larger than RAM. Read `CLAUDE.md` and `STATUS.md` first — `STATUS.md` is the
canonical scoreboard. Then read only the 2–3 graph nodes a task links to.

**The project was called `bigtea` until 2026-08-16.** Every crate, binary,
environment variable and document is now `chaos`; `bigtea-run` is `chaos-run`,
`BIGTEA_THREADS` is `CHAOS_THREADS`. If you find the old name anywhere, it is a
miss and should be fixed. The git remote is the one deliberate exception — Atur
renames the repository himself, and the `repository`/`homepage` URLs and the CI
badge already point at the new name.

## Where things stand

`main` is green: **566 tests, 0 failed**, workspace clippy clean, fmt clean.
**165 of llama.cpp's 182 long flags implemented, 17 declined with a written
reason, 0 unrecognised** — that count is computed from both binaries, not tallied
by hand, which is the only way it has ever been right. Re-audit the declined
table whenever you change what the engine can do; six of its reasons had rotted
by the third audit in one day.

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

All 2026-08-16, i7-13650HX / 15.7 GiB / RTX 3050, **both engines alternating in
one session**. Full node with every command line:
`research/where-we-stand-vs-llamacpp-2026-08-16.md`.

| | Chaos | llama.cpp | |
|---|---:|---:|---|
| V4-Flash prefill, ms/prompt token | 1640 | 1679 | parity |
| V4-Flash generation | 0.394 | 0.39 | parity |
| Qwen3-30B generation | 3.03–3.86 | 3.35 | parity, paired 3–2 |
| Qwen3-30B prefill | 1.22 | 1.17 | parity |
| Qwen3-4B generation, both at defaults | 8.01 | 6.52 ± 0.33 | 1.23x ahead |
| Qwen3-4B generation, both hand-tuned | 7.64 | 9.16 ± 0.43 | 1.20x behind |

- The old **"V4-Flash prefill 1.62x behind, generation 3-4x behind"** is
  **retracted**, and so is *"generation is ~2x behind on Qwen3-30B (1.07 vs
  2.16)"*. Do not quote either. **Do not claim a lead either** — the ranges
  overlap, and llama.cpp's discarded warm-up would have made V4-Flash look like a
  1.7x win.
- **The GPU is a dense-model dial.** On streaming MoE it is **4.3x slower**
  (2.61 → 0.61 tok/s) because the experts run on the host whatever `-ngl` says.
  `-ngl` on Qwen3-4B is 1.79x prefill / 1.40x generation, monotonic, no knee.
- **A V4-Flash token is 67% expert-slice read, 17% block compute, 16% routing**,
  and the whole routed expert arithmetic is **under 5%**.
- The drive tops out at **2.74 GiB/s at four handles** and does not climb at
  8/16/32.

**This machine drifts by more than most effects.** Anything worth under ~10%
needs both arms alternating in one session. Three wrong figures have been
published here by comparing against a remembered number.

**And stopping a benchmark does not stop the engine.** A `llama-completion`
orphaned by a killed wrapper sat holding 8.98 GiB and made every later run read
10x slow — which looks exactly like a catastrophic regression. Check for stray
processes before believing a surprising number.

## Dead ends — measured, do not re-propose

Expert factorisation · contextual sparsity · a pinned hot set · dropping the
router's tail · expert-read/compute overlap (1.03x) · `--op-offload` (19%
slower) · `mul_mat_id` batching on the streaming path · porting parallel-experts
to V4-Flash. **The byte-reduction roadmap is closed**: 20 tok/s needs 79
MB/token, V4-Flash reads 3288, and everything still alive multiplies to 3.1x
against a 42x gap.

## What is worth doing next

1. **The second machine** — RTX 5090 (32 GiB VRAM), 64 GiB RAM, modern Intel,
   SSD. 96 GiB of fast memory against 144 GiB of model is **~67% resident there
   against ~11% here**, so expect a materially different number and **measure it
   rather than predicting it**. Check `--auto` makes a sensible choice with the
   user knowing no flags, and fix it if not.
2. **Qwen3-30B-A3B is 17.3 GiB and fits entirely in a 5090's VRAM.** That is the
   demo. V4-Flash still streams there — 144 GiB does not fit in 96 — so do not
   promise coding-agent speed on it, on either machine.
3. **The tok/s-versus-RAM frontier for a 144 GB model.** Nobody has published it
   and only an engine that owns residency can sweep it; `mmap` cannot be told to
   use exactly N GiB. This is the honest product question: *given your machine,
   the largest model at the speed you want.*
4. **Verify the GPU tier.** It fails 1 of 8 parity prompts where the CPU path
   fails none. Shown to be arithmetic rather than wiring — the kernels disagree
   by 0.37–0.71 while the model's own top-2 margin falls to 0.399 — but still
   unproven either way, and it must not be called finished.
5. Finish R5/T1-T5 of `lts-0-0-0.md`: quant selection, self-configuration.

## Models on disk

```
C:\Projects\models\qwen3moe\Qwen3-30B-A3B-Q4_K_M.gguf      17.3 GiB   the demo
C:\Projects\models\qwen3-4b\Qwen3-4B-Q4_K_M.gguf            2.3 GiB   sanity check
C:\Projects\models\llama32-1b\Llama-3.2-1B-Instruct-...     0.7 GiB   parity harness
C:\Projects\models\v4flash\DeepSeek-V4-Flash-...00005      144.4 GiB  larger-than-RAM demo
```

Limited home internet — do not download more without asking.

## Rules that were each bought with a mistake

- **Never `git push -u`** — it writes the token into `.git/config`. Push with the
  token from `C:\Projects\.env` inline in the URL, output redacted, never echoed.
- Verify containment with `git merge-base --is-ancestor <branch> origin/main`
  *before* deleting a branch.
- After merging, fast-forward `main` explicitly and **re-run tests on `main`
  itself**. A local `main` with no upstream makes `git pull` a silent no-op —
  this session cloned a "fresh" checkout from the local repository and got a
  nine-commit-old tree that way. Clone from the remote when the point is to test
  what a stranger gets.
- GitHub parses only the *first* issue in `Closes #1, #2, #3`.
- **ggml aborts rather than returning errors** — an exhausted arena, an
  out-of-range device index, a misaligned host pointer. Check on the Rust side.
- **A wrong forward pass produces fluent nonsense, never a crash.** Loading is
  not evidence; answering in English is not evidence. Only a diff against
  llama.cpp counts.
- Correct Atur's English briefly at the top of every reply.
