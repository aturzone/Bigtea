---
topic: every trap in this codebase that cost real time, in full
status: reference
links:
  - ../research/parallel-experts-do-not-transfer-2026-08-16.md
  - ../research/v4flash-has-no-slack-2026-08-10.md
  - ../research/threads-were-never-plumbed-2026-08-10.md
  - ../research/where-we-stand-vs-llamacpp-2026-08-16.md
---

# Facts that cost time to rediscover

**This is the long form. `/CLAUDE.md` carries a one-line summary of each entry
here and points at this file.** It lived in `CLAUDE.md` until 2026-08-16, when
that file had grown to 3,308 words against its own ~2000-token budget — and a
budget nobody enforces is not a budget. Nothing was dropped in the move: every
sentence below was in `CLAUDE.md` verbatim, and the summary lines there are
lossy on purpose.

**Read this before proposing any optimisation.** Most of the appealing ideas
about this engine have already been tried, and about half of the entries below
are the measurement that killed one.

## ggml

- **ggml aborts** (`GGML_ASSERT`) when its arena is exhausted — no error to
  catch. Size arenas up front. **This also kills a whole test binary**: the 19
  V4-Flash tests each allocate GB-sized arenas and, run in parallel, exhausted
  memory and aborted the process — reported as `process didn't exit
  successfully`, not as a failing test, with every later result lost. They hold
  a shared `heavy()` lock now, so plain `--ignored` works.
- **Every arena must scale with the prefill block.** Fixed-size arenas abort
  once the block grows; ggml asks and dies rather than returning an error.
  **`available` in that message is the pool's total size, not the remainder** —
  read it as the remainder and you go looking at whichever arena was nearly full
  instead of the one that was too small. Divide `needed` by the tensor size
  instead: `56,624,208 ≈ 3 × 18,874,368` said "this arena budgeted one and
  allocated three" immediately. And **`arena_for` doubles its total, which hides
  an undercount until the block grows enough to eat it** — list every tensor a
  branch can allocate, for that branch.
- **ggml `ne[0]` is the fastest dimension.** Reading shapes as row-major
  transposes every matrix and yields confident nonsense.
- **Weights are bound zero-copy** (`no_alloc` + data pointer). A copy would need
  2× the model and not fit.
- **`compute()` re-evaluates the whole ancestor graph.** Calling it per
  intermediate *re-does* the work each time, plus a graph build and threadpool
  cycle. 24 calls per block became 6 — **1.9x**. Invisible on prefill (big
  matmuls bury it), dominant at one token. Compute only before a
  `to_vec_*`/`set_*`.
- **`compute(&t, 0)` runs on ONE thread** — the count is floored at 1, not
  defaulted to all cores. This silently ran every expert matmul
  single-threaded.
- **`flash_attn_ext` does NOT transpose V**, unlike the `mul_mat` attention
  path, and its mask must be **F16 and contiguous**. Both mistakes give fluent
  nonsense, not an error. Mask values are only 0 and -inf, so write the bits
  (`0x0000` / `0xFC00`) rather than converting.
- **`Arc::from(Box<[u8]>)` reallocates and copies** — hand `bind` the `Vec<u8>`
  instead (`WeightBytes` covers any `Deref<Target=[u8]>`); that mistake alone
  cost 12s of a 27s run.

## Correctness, which fails silently here

- A **wrong tokenizer or forward pass produces fluent nonsense**, never a crash.
  Test pieces separately.
- **Missing causal mask → repeated tokens**, not an error. Masked positions need
  `-inf`, not `0`.
- **top_k does not return indices in score order** — look expert weights up by
  index.
- **Router weights must be renormalised** over selected experts only.
- **Nothing in a GGUF records the FFN activation** — a GELU model and a SiLU
  model hold byte-identical tensor sets. The whole Gemma family is GELU and
  everything else here is SiLU; the wrong one is not a missing tensor, not a
  shape error and not a crash, just a model that answers fluently and disagrees
  with llama.cpp from the first token. `gemma2` sat in `VERIFIED_ARCHITECTURES`
  in that state for weeks. **Membership in that list means someone ran the
  reference — loading is not evidence and answering in English is not
  evidence.**
- **Match the reference's *order*, not its algebra, wherever a soft cap is
  involved.** llama.cpp pre-scales Q and passes `scale = 1.0`; ggml folds the
  cap into the scale (`scale /= cap`), so passing the scale instead is the same
  arithmetic and `0.0625f/50f` vs `0.0625f*(1f/50f)` differ by **one ULP**.
  Through `tanh` that flipped Gemma-2's first token and rewrote the whole
  completion. A cap turns a scale into a non-linearity's argument, and then the
  last bit is not decorative.
- **`chaos-run -v` prints the derived hparams** (`attn_scale`, per-layer RoPE
  bases, windowed-layer list). Use it before theorising: a key read under the
  wrong name looks exactly like a key that was absent.
- **Prompt length decides which code paths run.** V4-Flash's compressed
  attention builders are guarded on their caches being non-empty, so the *same
  layer* runs different attention at different lengths: at 2 tokens all 43
  blocks fall back to the Raw path, at 5 CSA fires, at 165 HCA fires, and the
  sparse indexer selects nothing until >2048. A shorter capture can reach
  *further* than a longer one. See `../research/v4flash-compressed-attention.md`.
- **Routing is not bitwise stable across sequence lengths.** At 63 → 64 tokens
  the *same* earlier tokens re-routed ~3% of their selections (net still +6 per
  layer, so nothing was lost) — near-ties in the top-6-of-256 flipping when the
  batch shape changes. Layers 0-2 (token-id routed) were untouched, so it
  arrives through attention. **Mechanism unidentified**: "ggml re-blocks at
  multiples of 64" was the first guess and a 166→212 run crossing 192 showed
  zero churn, so it is not that. A test demanding equal routing across batch
  shapes will fail on correct code.

## Residency and streaming

- **Expert access is a cyclic scan, so recency-based caching is the worst policy
  available.** Layer 0 is always the oldest entry when layer 47 needs room.
  Frequency-gated admission took hit rate 17% → 70% at the same budget.
- **Expert reads are deduplicated per block across the whole batch.** A pass
  reads the *distinct* experts its tokens select, not one slice per selection
  (`read_expert_slices` takes `unique`). Measured distinct experts per layer per
  pass: **6 at one token (3.2 GiB), 39.7 at 17 tokens (21 GiB), 122.8 at 166
  tokens (66 GiB)** — selections per layer grow 10x from 17 to 166 tokens while
  distinct reads only grow 3x. **So a cache's value depends on how many distinct
  experts a step touches, not on how skewed routing is**, and only a KV-cached
  single-token step is small enough for a few GiB to cover.
- **Cache hit rate is not a success metric.** Past ~6 GiB the expert cache
  reaches 71% hits and is the *slowest* configuration measured: cached bytes get
  paged out, so a "hit" is a page fault wearing a disguise. Only tok/s at a
  stated footprint counts.
  - **Partly retracted 2026-08-16**: the *slowest* half does not reproduce on
    Qwen3-30B, where a 2/4/6/8 GiB sweep gives 2.22/2.66/3.45/3.43 tok/s — it
    plateaus at 6 GiB rather than declining, and the default already sits on the
    plateau (`../research/expert-read-overlap-does-not-pay-2026-08-16.md`). The
    headline stands and the mechanism stands; "more cache eventually goes
    backwards" is a V4-Flash observation that was over-generalised.
- **GGUF pads tensor data to `general.alignment` (32), not to a disk sector.**
  So tensors start mid-sector and a conventionally *aligned* buffer can never
  receive a direct transfer — every byte bounces. Skew the destination to
  `file_offset % 4096` instead (`SkewedBuf`): 0.80 → 1.58 GiB/s, 0.09% copied.
- **Concurrent readers need a file handle EACH.** A Windows handle without
  `FILE_FLAG_OVERLAPPED` is synchronous and the OS serialises reads on it, so N
  threads on one handle hold the drive at queue depth 1. The old "no gain past 4
  readers, the drive does 2.37 GiB/s" was this artefact: same reads, 4 threads,
  **2.01 GiB/s shared vs 2.65 per-handle**, and per-handle beats the "sequential
  ceiling". `Shard` now pools 8 handles.
- **The drive tops out at 2.74 GiB/s and stops climbing at FOUR handles**
  (`chaos-iobench`, 4 MiB scattered slices; 8/16/32 do not improve on it). So
  the 8-handle pool is not the limit — the gap between that and V4-Flash's
  achieved 1.88 GiB/s is the per-block barrier, and nothing can be queued during
  it because the next block's addresses depend on routing it has not computed
  yet.

## Threads

- **Threads are two levers pulling opposite ways, and `-t` reached only one
  architecture.** Generation saturates DRAM and wants **2-4** threads; prefill
  is compute-bound and wants **all** of them (Qwen3-4B: gen 7.64 @2 vs 4.49 @20;
  prefill 47.4 @4 vs 81.5 @20). Hence `-t` *and* `-tb`, picked by the step's
  token count. The old "threads are not the lever" reading came from a sweep
  whose knob was disconnected — `-t` set `CHAOS_THREADS`, which only
  `deepseek4_forward.rs` read, so `-t 1` and `-t 20` gave *bit-identical* phase
  timings. **A disconnected knob is indistinguishable from a flat response;
  check the knob moves something first.** Fixing it was 1.66x/1.69x.
- **V4-Flash needs the same split, and the old "threads are not the lever" note
  was measured too short.** At 5 tokens a V4-Flash prefill is almost all disk,
  so 4/12/20 did cost the same; **at 180 tokens it is 2.24 (4 threads) against
  2.89 (all)**. Generation is the opposite — `-t 4` beat `-t 20` in two
  back-to-back sessions, 0.380/0.296 and 0.196/0.177. **Absolute V4-Flash
  numbers drift a lot with page-cache state; only compare within one session.**
- **The MoE expert path wants ONE thread — 2.4x on Qwen3-30B** (2.88 tok/s at 1
  vs 1.21 at 20; expert compute 2.2s → 5.2s). A layer's graph holds 24
  matrix-vector products of 768x2048; split 20 ways that is ~38 rows per thread
  per barrier, and the threads cost more than the work. **llama.cpp peaks at 4
  threads where we peak at 1**, because ggml parallelises *within* a node and 38
  rows per thread is not worth a barrier. **Closed 2026-08-16 from the other
  side**: parallelise ACROSS experts — N whole subgraphs, one ggml thread each,
  summed in Rust — 1.29x on expert compute and 1.10x end to end, output
  byte-identical. Nothing is gathered, so the ~1.02 GB/token that killed the
  `mul_mat_id` route never appears
  (`../research/parallel-experts-2026-08-16.md`).

## Measurement

- **Profile before optimising a streaming runner.** The largest cost in
  generation was memcpy — slices copied twice per use — not disk and not
  arithmetic. Nothing suggested it until it was timed.
- **A hot set scored on the prompt it was chosen from tells you nothing.** "64
  experts absorb 97.8% of selections" was in-sample on one prompt; out of sample
  it is 53.7%, and 37.5% across subjects against 25% for caching at random.
  Always score a residency policy on data it did not see. Two matching controls
  are cheap and both were missing: a **uniform null at the same sample size**
  (with ~1000 draws over 256 experts, top-64 covers 41% by construction) and a
  **noise ceiling** (resample the same distribution — if cross-prompt sits below
  it, the divergence is real).
- **Statistics computed over `chaos-run`'s output double-count.** Regeneration
  is stateless, so every generated token re-runs prefill and the routing
  histogram counts the same prompt again: chi-square went 1282 → 5464 → 11469
  for 1/4/8 tokens while coverage never moved. Capture with `-n 1`.
- **Do not calibrate on a proxy.** A 150 ms DRAM-saturation benchmark picked
  6/8/12/12/4/6 on six identical runs while the true optimum was 2-4, and its
  spread was worse than the bad default it replaced — a pure read has no
  per-node barrier, a ggml graph does. Tune on real generated tokens instead. A
  proxy corrected until it agrees with the objective *is* the objective,
  measured badly.
- **A counter inside an overlapped path measures the overlap, not the work.**
  The obvious way to price a residency shortfall was to accumulate bytes and
  elapsed time in `prefetch_dense_via`, the funnel every spilled read passes
  through. It reads **0.80 GiB/s** against a swept truth of 2.44, because R2
  overlap runs that prefetch on 2 of 8 handles for the whole duration of a
  block — its wall clock is how long the thread was *occupied*.
  `CHAOS_PREFETCH_OVERLAP=0` reads 1.99 on the same binary. Built, measured,
  reverted; the same shape as the `dense` phase timer reading 0.01 s per token
  while the spill demonstrably costs 0.41 s/GiB.
- **The load rate is not the re-read rate, and the difference is queue depth.**
  `chaos-run` priced a shortfall at `missing / LoadReport::bytes_per_sec()` and
  overstated it by ~1.5x for two years' worth of sessions: the load is
  essentially one stream at 1.6-2.0 GB/s, while the spill comes back across the
  eight-handle pool at 2.4-2.7 GiB/s. What ships re-reads a sample of **the
  spilled tensors themselves** through the same pool — the operation, not a model
  of it. Its sizing had to be measured too: capping each read at 16 MiB swung the
  answer 1.54-2.65 GiB/s, because whether a tensor exceeded the cap changed the
  read size.
- **A kernel benchmark measures the kernel, not the data movement needed to feed
  it.** `chaos-kernelbench` put the batched `mul_mat_id` expert form at 11.17
  GiB/s with 2.86x thread scaling — real, but it binds the model's
  *already-stacked* tensor zero-copy. On the streaming path the selected experts
  are unrelated `Arc<[u8]>`, and making them contiguous costs ~1.02 GB/token,
  which is what the kernel saves. Built, byte-identical output, **1.34 → 1.27
  tok/s, reverted.** The version that pays needs the experts resident.
- **Killing a benchmark's wrapper does not kill the engine, and an orphan is
  invisible in the numbers.** A stopped background script left `llama-completion`
  alive holding **8.98 GiB**; every run after it read 10x slow (V4-Flash
  generation 0.039 against 0.39) and looked exactly like a regression.
  `Get-Process` before trusting a surprising number, and prefer letting a
  comparison finish over stopping it.
- **A competitive claim is not citable until the competitor's exact command line
  and its output are in a doc.** "llama.cpp can't do X" survived days on a
  misattributed error string because nobody ran the opposing command. Run it,
  paste it, flag it.
- **And it needs REPEATS, because the first run of a GPU path is a different
  program from the second.** ggml's Vulkan backend compiles a large shader set
  on first use and the driver persists the pipelines to disk, so run 1 pays
  compilation *inside the timed region*. That published "the card is 0.42x the
  CPU" with a confident causal story about PCIe round trips; the same binary
  then measured 1.49x, then 1.6-1.8x. **Discard the first run.** Three failures
  there, only one of which was the number: a cold-cache run reported as steady
  state, a mechanism asserted rather than measured (1.4 GB moves in under a
  second, against a ten-second gap — the arithmetic contradicted the story and
  nobody checked), and a retraction found *by accident* when a build failed and
  the old binary ran again. **Nothing re-measures a number already written
  down**, so the guard has to be in the harness.

## V4-Flash specifically

- **V4-Flash has no redundancy left to harvest — four probes, four negatives.**
  Experts are 9.1% internally negligible; the expert *bank* is full-rank (a
  rank-512 shared basis holds 20.4% of its energy against **16.6% for random
  noise**, `chaos-spectrum`); the router's tail is not small
  (33.5/20.6/15.0/12.1/10.1/**8.8**%, so 3-of-6 discards 31% of the mass); and a
  pinned hot set scores 37.5% vs 25.0% random. **3.21 GiB/token is what the
  model costs, not an artefact.** Do not re-propose factorisation, contextual
  sparsity, or pinning.
- **The expert matmul is a few percent of a token on V4-Flash, and the
  parallel-experts win does NOT port there.** 3.02 ms per block at 24.7 GiB/s —
  above single-threaded memcpy, i.e. already at DRAM speed. Measured directly
  2026-08-16 by dropping the three routed `mul_mat_id` calls and keeping the
  read: generation **0.388 against 0.370**, block `compute` **0.01s of 0.44** —
  so the whole routed arithmetic is **under 5%** and perfect parallelisation is
  worth at most 1.05x. **A V4-Flash token is 67% expert-slice read, 17% block
  compute, 16% routing.** There is also nothing to gather: `read_expert_slices`
  packs the slices contiguously as it reads them, so this path already runs the
  batched form for free. Compute scales as ~`n^0.49` in the batch, so
  batched/speculative passes are cheaper than a linear model predicts
  (`../research/parallel-experts-do-not-transfer-2026-08-16.md`).
- **Speculative decoding is ~1.4x here, not 2.2x.** The literature assumes the
  verify pass costs what a single-token pass costs; here it costs more, because
  more tokens select more distinct experts (`U(n)≈6·n^0.667`). Below α≈0.75 it
  is a net *loss*, and the optimum draft is short.

## Toolchain

- **Windows: `.cargo/config.toml` sets `link-self-contained=no`.** MSYS2 gcc
  16.1.0 dropped symbols rustup's bundled `crt2.o` still references, so every
  link fails with "undefined reference" on code that compiles. Do not delete it.
- Windows needs the **GNU** Rust toolchain
  (`rustup default stable-x86_64-pc-windows-gnu`) plus MSYS2 mingw64 on PATH.
  `[[bin]]` targets set `test = false` — empty harnesses are pointless and Smart
  App Control blocks unsigned fresh binaries.

## The window

Every entry here cost a rebuild and a screenshot. **A GUI is not verified by
compiling**, and three of these were believed fixed before a pixel was measured.

- **Never hold a `RefCell` borrow across a call Windows can re-enter.**
  `SendMessageW`, `EnableWindow`, `SetWindowTextW`, `MoveWindow`, `ShowWindow`
  and `SetFocus` can all dispatch `WM_CTLCOLOR*` synchronously, which borrows
  the same cell. Under `panic = "abort"` the double borrow is instant, silent
  process death — no message, no log, no stack. Pull handles and data *out* of
  the borrow, drop it, then talk to Windows. `tests/ui_rules.rs` enforces this
  textually, and found three more instances the day it was written.
- **Find controls by id, not by storing handles in the state.** `GetDlgItem`
  needs only the window handle, which can live in an atomic — so no action
  function needs a borrow open in order to locate a control. This removes the
  bug above by construction rather than by discipline.
- **A `thread_local!` is invisible to worker threads.** `notify()` read the UI
  handle from one, saw `None` on every worker, posted nothing, and every
  generated token was received and discarded while the status line said
  "ready". Anything a background thread needs goes in an atomic or a `Mutex`.
- **A read-only `EDIT` silently ignores `EM_REPLACESEL`** — no error, no text.
  Clear `ES_READONLY`, append, set it again.
- **`GetWindowText` returns empty for a control owned by another process**, so
  a cross-process UI test reads every box as empty whether the app works or
  not. This cost an hour and nearly produced a fix for a bug that did not
  exist. `WM_GETTEXT` sent with `SendMessage` *does* work across processes;
  otherwise a screenshot is the evidence.
- **Windows draws the menu bar, and it does not follow dark mode.**
  `DwmSetWindowAttribute(DWMWA_USE_IMMERSIVE_DARK_MODE)` darkens the title bar
  only. `SetPreferredAppMode` (uxtheme ordinal 135, with `FlushMenuThemes` at
  136) is what every dark Win32 app calls; on 10.0.26200 the ordinals resolve,
  the call runs, and the bar still measures `#FFFFFF` — tried both before and
  after window creation. **It was removed rather than shipped as a no-op.**
  Scrollbars *are* fixable: `SetWindowTheme(h, "DarkMode_Explorer", NULL)` on
  each control moves them from `#F0F0F0` to `#171717`, and that call alone is
  sufficient — the app-mode call contributed nothing to it either.
- **Owner-draw is the only way to colour a button or a list selection.** A
  themed push button ignores `WM_CTLCOLORBTN` entirely and the selection bar is
  the system highlight.
- **`ne`-style geometry belongs in one function.** The settings page positions
  its boxes in `layout` and draws their labels in `paint`; two independent walks
  of the same list is how a label lands over the wrong box. One function returns
  the run, both callers use it.
- **Do not do file I/O while painting.** Counting a model's shards in the
  detail panel meant a directory scan per repaint, and the transcript repaints
  on every token. Count it once, in the rescan.

## The installer

- **`Vec::as_ptr()` on an empty vector is a dangling pointer, and Windows will
  dereference it.** `DrawTextW` with a zero-length buffer took the installer
  down the instant its report reached a blank line: a stack-cookie failure
  (`c0000409`), not an access violation, so it did not even look like a null
  dereference. Guard every text call on `!is_empty()`.
- **A panic inside `extern "system"` never reaches the panic hook.** Unwinding
  out of a non-`C-unwind` function is undefined, so Rust aborts at the boundary
  — the hook does not run, no log is written, and the window simply vanishes.
  A `wndproc` is exactly such a function, so nothing that happens during
  painting can report itself. The only way to find one is to log through it.
- **The Windows Application event log is the last resort and it does help.**
  `Get-EventLog -LogName Application` gave the fastfail code and the faulting
  offset when there was no crash file at all, which is what said "Rust abort"
  rather than "access violation" and ruled out half the candidates.
- **An installer needs a log more than any other program here.** It runs once,
  on a machine that is not yours, and the person running it cannot rerun it
  under a debugger. `%TEMP%\chaos-setup.log` is written a line at a time,
  opened and closed per line so an abort cannot lose a buffered write — and it
  is what found the crash above in one run.
- **Do the work on a worker, not in the message loop.** The old install ran
  inside `WM_COMMAND`, so the window was frozen for its whole duration and said
  nothing about what it was doing. Every step now reports before and after, and
  the list shows which one is in flight.
- **`CreateFontW` never fails.** Ask for a face that is not installed and GDI
  substitutes silently, so a display serif chosen for a wordmark quietly becomes
  the UI font. Select it into a DC and ask `GetTextFaceW` what actually came
  back; `first_available_face` does this.
