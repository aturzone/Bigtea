# Chaos — a runner for models larger than RAM

> **Read `STATUS.md` first.** It is the canonical statement of where the project
> is, the honest scoreboard, and what remains in order. Update it in the same
> commit as anything that moves a number or closes a task.

- **What it is**: a Rust inference runner whose job is running models that do *not* fit in memory. Keeps the always-read weights resident, streams routed experts from disk per token. Borrows `ggml` for arithmetic; owns memory, residency, streaming, and the token loop.
- **Proven**: Qwen3-30B-A3B (17.28 GiB container) generates correct text on a 15.7 GiB machine holding 0.93 GiB resident + a 6.26 GiB expert cache.
- **Prefill beats llama.cpp** at 565 (27.6 vs 23.6) and 2206 tokens (36.6 vs 33.6), and matches it at 4395 and 8775; `-b 4096` gives 43.6 vs 40.3. **Generation was ~2x behind (1.07 vs 2.16); re-measured 2026-08-16 it is 0.90x** (3.03 vs 3.35, five alternating pairs) — still behind, now inside the noise. The 2x figure is dead; do not quote it, and do not claim a lead either (`qwen3moe-generation-parity-2026-08-16.md`). llama.cpp also runs the 144 GB V4-Flash once `--no-repack` is passed, so "larger than RAM" is not a differentiator. Full ladder, retracted claims, and one experiment that failed: `docs/graph/research/head-to-head-llamacpp-2026-08-05.md`.
- Graph docs live in `/docs/graph/`; read `INDEX.md` first, then only the 2–3 nodes a task links to.

## Build / test / run

```
# ggml must be built first; point GGML_LIB_DIR at ggml-base.a, ggml-cpu.a, ggml.a
export GGML_LIB_DIR=C:/Projects/llamacpp-unsloth/build/ggml/src   # PowerShell: $env:GGML_LIB_DIR=...
# GPU work needs build-vulkan/ggml/src instead. That build above has NO Vulkan
# archive, and the GPU tests SKIP rather than fail without a card -- so a green
# "6 passed" was reported for a file whose two GPU tests never ran once.
cargo test --release          # 224 tests
cargo test --release --test deepseek4_forward -- --ignored   # 19 V4-Flash, needs the container
cargo build --release
./target/release/chaos-run <model.gguf> "prompt" -n 16
./target/release/chaos-probe --quick          # RAM/disk/GPU + what to close
./target/release/chaos-model-info <m.gguf> --budget 8   # fit + tok/s prediction
```

Windows: needs the **GNU** Rust toolchain (`rustup default stable-x86_64-pc-windows-gnu`) plus MSYS2 mingw64 on PATH. `[[bin]]` targets set `test = false` — empty harnesses are pointless and Smart App Control blocks unsigned fresh binaries.

## Crates

`gguf` container parsing · `probe` hardware + RAM reclaim · `plan` prediction + residency policy · `io` cache-bypassing aligned reads · `model` sharded resolution + partial reads · `ggml` FFI (graph, zero-copy weight binding) · `tokenizer` byte-level BPE · `arch` architectures + streaming forward pass

## Facts that cost time to rediscover

- **ggml aborts** (`GGML_ASSERT`) when its arena is exhausted — no error to catch. Size arenas up front. **This also kills a whole test binary**: the 19 V4-Flash tests each allocate GB-sized arenas and, run in parallel, exhausted memory and aborted the process — reported as `process didn't exit successfully`, not as a failing test, with every later result lost. They hold a shared `heavy()` lock now, so plain `--ignored` works.
- **ggml `ne[0]` is the fastest dimension.** Reading shapes as row-major transposes every matrix and yields confident nonsense.
- **Weights are bound zero-copy** (`no_alloc` + data pointer). A copy would need 2× the model and not fit.
- **Missing causal mask → repeated tokens**, not an error. Masked positions need `-inf`, not `0`.
- **top_k does not return indices in score order** — look expert weights up by index.
- **Router weights must be renormalised** over selected experts only.
- A **wrong tokenizer or forward pass produces fluent nonsense**, never a crash. Test pieces separately.
- **`compute(&t, 0)` runs on ONE thread** — the count is floored at 1, not defaulted to all cores. This silently ran every expert matmul single-threaded.
- **Expert access is a cyclic scan, so recency-based caching is the worst policy available.** Layer 0 is always the oldest entry when layer 47 needs room. Frequency-gated admission took hit rate 17% → 70% at the same budget.
- **Profile before optimising a streaming runner.** The largest cost in generation was memcpy — slices copied twice per use — not disk and not arithmetic. Nothing suggested it until it was timed.
- **Expert reads are deduplicated per block across the whole batch.** A pass reads the *distinct* experts its tokens select, not one slice per selection (`read_expert_slices` takes `unique`). Measured distinct experts per layer per pass: **6 at one token (3.2 GiB), 39.7 at 17 tokens (21 GiB), 122.8 at 166 tokens (66 GiB)** — selections per layer grow 10x from 17 to 166 tokens while distinct reads only grow 3x. **So a cache's value depends on how many distinct experts a step touches, not on how skewed routing is**, and only a KV-cached single-token step is small enough for a few GiB to cover.
- **Routing is not bitwise stable across sequence lengths.** At 63 → 64 tokens the *same* earlier tokens re-routed ~3% of their selections (net still +6 per layer, so nothing was lost) — near-ties in the top-6-of-256 flipping when the batch shape changes. Layers 0-2 (token-id routed) were untouched, so it arrives through attention. **Mechanism unidentified**: "ggml re-blocks at multiples of 64" was the first guess and a 166→212 run crossing 192 showed zero churn, so it is not that. A test demanding equal routing across batch shapes will fail on correct code.
- **A hot set scored on the prompt it was chosen from tells you nothing.** "64 experts absorb 97.8% of selections" was in-sample on one prompt; out of sample it is 53.7%, and 37.5% across subjects against 25% for caching at random. Always score a residency policy on data it did not see. Two matching controls are cheap and both were missing: a **uniform null at the same sample size** (with ~1000 draws over 256 experts, top-64 covers 41% by construction) and a **noise ceiling** (resample the same distribution — if cross-prompt sits below it, the divergence is real).
- **Statistics computed over `chaos-run`'s output double-count.** Regeneration is stateless, so every generated token re-runs prefill and the routing histogram counts the same prompt again: chi-square went 1282 → 5464 → 11469 for 1/4/8 tokens while coverage never moved. Capture with `-n 1`.
- **Cache hit rate is not a success metric.** Past ~6 GiB the expert cache reaches 71% hits and is the *slowest* configuration measured: cached bytes get paged out, so a "hit" is a page fault wearing a disguise. Only tok/s at a stated footprint counts.
- **`flash_attn_ext` does NOT transpose V**, unlike the `mul_mat` attention path, and its mask must be **F16 and contiguous**. Both mistakes give fluent nonsense, not an error. Mask values are only 0 and -inf, so write the bits (`0x0000` / `0xFC00`) rather than converting.
- **Prompt length decides which code paths run.** V4-Flash's compressed attention builders are guarded on their caches being non-empty, so the *same layer* runs different attention at different lengths: at 2 tokens all 43 blocks fall back to the Raw path, at 5 CSA fires, at 165 HCA fires, and the sparse indexer selects nothing until >2048. A shorter capture can reach *further* than a longer one. See `v4flash-compressed-attention.md`.
- **GGUF pads tensor data to `general.alignment` (32), not to a disk sector.** So tensors start mid-sector and a conventionally *aligned* buffer can never receive a direct transfer — every byte bounces. Skew the destination to `file_offset % 4096` instead (`SkewedBuf`): 0.80 → 1.58 GiB/s, 0.09% copied.
- **`compute()` re-evaluates the whole ancestor graph.** Calling it per intermediate *re-does* the work each time, plus a graph build and threadpool cycle. 24 calls per block became 6 — **1.9x**. Invisible on prefill (big matmuls bury it), dominant at one token. Compute only before a `to_vec_*`/`set_*`.
- **Threads are two levers pulling opposite ways, and `-t` reached only one architecture.** Generation saturates DRAM and wants **2-4** threads; prefill is compute-bound and wants **all** of them (Qwen3-4B: gen 7.64 @2 vs 4.49 @20; prefill 47.4 @4 vs 81.5 @20). Hence `-t` *and* `-tb`, picked by the step's token count. The old "threads are not the lever" reading came from a sweep whose knob was disconnected — `-t` set `CHAOS_THREADS`, which only `deepseek4_forward.rs` read, so `-t 1` and `-t 20` gave *bit-identical* phase timings. **A disconnected knob is indistinguishable from a flat response; check the knob moves something first.** Fixing it was 1.66x/1.69x.
- **V4-Flash needs the same split, and the old "threads are not the lever" note was measured too short.** At 5 tokens a V4-Flash prefill is almost all disk, so 4/12/20 did cost the same; **at 180 tokens it is 2.24 (4 threads) against 2.89 (all)**. Generation is the opposite — `-t 4` beat `-t 20` in two back-to-back sessions, 0.380/0.296 and 0.196/0.177. **Absolute V4-Flash numbers drift a lot with page-cache state; only compare within one session.**
- **The MoE expert path wants ONE thread — 2.4x on Qwen3-30B** (2.88 tok/s at 1 vs 1.21 at 20; expert compute 2.2s → 5.2s). A layer's graph holds 24 matrix-vector products of 768x2048; split 20 ways that is ~38 rows per thread per barrier, and the threads cost more than the work. **llama.cpp peaks at 4 threads where we peak at 1**, because ggml parallelises *within* a node and 38 rows per thread is not worth a barrier. **Closed 2026-08-16 from the other side**: parallelise ACROSS experts — N whole subgraphs, one ggml thread each, summed in Rust — 1.29x on expert compute and 1.10x end to end, output byte-identical. Nothing is gathered, so the ~1.02 GB/token that killed the `mul_mat_id` route never appears (`parallel-experts-2026-08-16.md`).
- **A kernel benchmark measures the kernel, not the data movement needed to feed it.** `chaos-kernelbench` put the batched `mul_mat_id` expert form at 11.17 GiB/s with 2.86x thread scaling — real, but it binds the model's *already-stacked* tensor zero-copy. On the streaming path the selected experts are unrelated `Arc<[u8]>`, and making them contiguous costs ~1.02 GB/token, which is what the kernel saves. Built, byte-identical output, **1.34 → 1.27 tok/s, reverted.** The version that pays needs the experts resident. Also: `Arc::from(Box<[u8]>)` **reallocates and copies** — hand `bind` the `Vec<u8>` instead (`WeightBytes` covers any `Deref<Target=[u8]>`); that mistake alone cost 12s of a 27s run.
- **Do not calibrate on a proxy.** A 150 ms DRAM-saturation benchmark picked 6/8/12/12/4/6 on six identical runs while the true optimum was 2-4, and its spread was worse than the bad default it replaced — a pure read has no per-node barrier, a ggml graph does. Tune on real generated tokens instead. A proxy corrected until it agrees with the objective *is* the objective, measured badly.
- **Concurrent readers need a file handle EACH.** A Windows handle without `FILE_FLAG_OVERLAPPED` is synchronous and the OS serialises reads on it, so N threads on one handle hold the drive at queue depth 1. The old "no gain past 4 readers, the drive does 2.37 GiB/s" was this artefact: same reads, 4 threads, **2.01 GiB/s shared vs 2.65 per-handle**, and per-handle beats the "sequential ceiling". `Shard` now pools 8 handles.
- **The expert matmul is a few percent of a token on V4-Flash, and the parallel-experts win does NOT port there.** 3.02 ms per block at 24.7 GiB/s — above single-threaded memcpy, i.e. already at DRAM speed. Measured directly 2026-08-16 by dropping the three routed `mul_mat_id` calls and keeping the read: generation **0.388 against 0.370**, block `compute` **0.01s of 0.44** — so the whole routed arithmetic is **under 5%** and perfect parallelisation is worth at most 1.05x. **A V4-Flash token is 67% expert-slice read, 17% block compute, 16% routing.** There is also nothing to gather: `read_expert_slices` packs the slices contiguously as it reads them, so this path already runs the batched form for free. Compute scales as ~`n^0.49` in the batch, so batched/speculative passes are cheaper than a linear model predicts (`parallel-experts-do-not-transfer-2026-08-16.md`).
- **Every arena must scale with the prefill block.** Fixed-size arenas abort once the block grows; ggml asks and dies rather than returning an error. **`available` in that message is the pool's total size, not the remainder** — read it as the remainder and you go looking at whichever arena was nearly full instead of the one that was too small. Divide `needed` by the tensor size instead: `56,624,208 ≈ 3 × 18,874,368` said "this arena budgeted one and allocated three" immediately. And **`arena_for` doubles its total, which hides an undercount until the block grows enough to eat it** — list every tensor a branch can allocate, for that branch.
- **V4-Flash has no redundancy left to harvest — four probes, four negatives.** Experts are 9.1% internally negligible; the expert *bank* is full-rank (a rank-512 shared basis holds 20.4% of its energy against **16.6% for random noise**, `chaos-spectrum`); the router's tail is not small (33.5/20.6/15.0/12.1/10.1/**8.8**%, so 3-of-6 discards 31% of the mass); and a pinned hot set scores 37.5% vs 25.0% random. **3.21 GiB/token is what the model costs, not an artefact.** Do not re-propose factorisation, contextual sparsity, or pinning.
- **Speculative decoding is ~1.4x here, not 2.2x.** The literature assumes the verify pass costs what a single-token pass costs; here it costs more, because more tokens select more distinct experts (`U(n)≈6·n^0.667`). Below α≈0.75 it is a net *loss*, and the optimum draft is short.
- **Nothing in a GGUF records the FFN activation** — a GELU model and a SiLU model hold byte-identical tensor sets. The whole Gemma family is GELU and everything else here is SiLU; the wrong one is not a missing tensor, not a shape error and not a crash, just a model that answers fluently and disagrees with llama.cpp from the first token. `gemma2` sat in `VERIFIED_ARCHITECTURES` in that state for weeks. **Membership in that list means someone ran the reference — loading is not evidence and answering in English is not evidence.**
- **Match the reference's *order*, not its algebra, wherever a soft cap is involved.** llama.cpp pre-scales Q and passes `scale = 1.0`; ggml folds the cap into the scale (`scale /= cap`), so passing the scale instead is the same arithmetic and `0.0625f/50f` vs `0.0625f*(1f/50f)` differ by **one ULP**. Through `tanh` that flipped Gemma-2's first token and rewrote the whole completion. A cap turns a scale into a non-linearity's argument, and then the last bit is not decorative.
- **`chaos-run -v` prints the derived hparams** (`attn_scale`, per-layer RoPE bases, windowed-layer list). Use it before theorising: a key read under the wrong name looks exactly like a key that was absent.
- **Killing a benchmark's wrapper does not kill the engine, and an orphan is invisible in the numbers.** A stopped background script left `llama-completion` alive holding **8.98 GiB**; every run after it read 10x slow (V4-Flash generation 0.039 against 0.39) and looked exactly like a regression. `Get-Process` before trusting a surprising number, and prefer letting a comparison finish over stopping it.
- **The drive tops out at 2.74 GiB/s and stops climbing at FOUR handles** (`chaos-iobench`, 4 MiB scattered slices; 8/16/32 do not improve on it). So the 8-handle pool is not the limit — the gap between that and V4-Flash's achieved 1.88 GiB/s is the per-block barrier, and nothing can be queued during it because the next block's addresses depend on routing it has not computed yet.
- **Windows: `.cargo/config.toml` sets `link-self-contained=no`.** MSYS2 gcc 16.1.0 dropped symbols rustup's bundled `crt2.o` still references, so every link fails with "undefined reference" on code that compiles. Do not delete it.

## Working rules

- Git: remote `github.com/aturzone/Chaos`. Push with the token from `C:\Projects\.env` inline in the URL, output redacted — never in git config, never echoed. Model/weight files stay gitignored.
- Implementation goes on `ticket/<name>` branches + PR. **Claude owns git end to end**: merge when CI is green, close what it supersedes, delete the branch, prune, and leave `main` verified. Docs may go to main.
- **Git hygiene, each rule bought with a mistake.** Verify containment with `git merge-base --is-ancestor <branch> origin/main` *before* deleting, never from "it was merged". After merging, `git checkout main` is not enough — a local `main` with no upstream makes `git pull` a silent no-op and leaves a pre-merge tree; fast-forward from `origin/main` explicitly and check a file that only the merge added. Then **re-run tests on `main` itself**, not on the branch. GitHub parses only the *first* issue in `Closes #1, #2, #3`, so give every one its own `closes`. Never `git push -u` — it writes the token into `.git/config`.
- Sync audit at phase boundaries only, not per commit.
- **A competitive claim is not citable until the competitor's exact command line and its output are in a doc.** "llama.cpp can't do X" survived days on a misattributed error string because nobody ran the opposing command. Run it, paste it, flag it.
- **And it needs REPEATS, because the first run of a GPU path is a different program from the second.** ggml's Vulkan backend compiles a large shader set on first use and the driver persists the pipelines to disk, so run 1 pays compilation *inside the timed region*. That published "the card is 0.42x the CPU" with a confident causal story about PCIe round trips; the same binary then measured 1.49x, then 1.6-1.8x. **Discard the first run.** Three failures there, only one of which was the number: a cold-cache run reported as steady state, a mechanism asserted rather than measured (1.4 GB moves in under a second, against a ten-second gap — the arithmetic contradicted the story and nobody checked), and a retraction found *by accident* when a build failed and the old binary ran again. **Nothing re-measures a number already written down**, so the guard has to be in the harness.
- Keep this file under ~2000 tokens; tell Atur to prune rather than letting it bloat.

## Next

**v0.0.2 released 2026-08-07. Renamed to `chaos` 2026-08-16**, and the release
for that name is what this repository is being cut for. `STATUS.md` is the
scoreboard; read `backlog/next-session-handoff.md` for the work queue.

**Against llama.cpp, measured 2026-08-16 with both engines alternating**
(`where-we-stand-vs-llamacpp-2026-08-16.md`): **parity on everything that
streams** — V4-Flash prefill 1640 against 1679 ms/prompt token and generation
0.394 against 0.39, Qwen3-30B parity on both phases. Behind by 1.20-1.27x on the
dense path when both sides are hand-tuned; ahead 1.23x out of the box, because we
measure the machine and llama.cpp uses a fixed default. **The old "V4-Flash
prefill 1.62x behind, generation 3-4x behind" is retracted.** Do not replace it
with a claimed lead either: the ranges overlap.

**The byte-reduction roadmap is closed** (`v4flash-has-no-slack-2026-08-10.md`).
20 tok/s needs 79 MB/token; V4-Flash reads 3288. Everything still alive
multiplies to 3.1x against a 42x gap. **20 tok/s is not a code problem** — it
needs the active weights to stop coming from disk.

1. **The tok/s-versus-RAM frontier for a 144 GB model** — nobody has published
   it, and only an engine that owns residency can sweep it (`mmap` cannot be told
   to use exactly N GiB). Answers the product question honestly: *given your
   machine, the largest model at the speed you want.*
2. **The 5090 machine** (32 GiB VRAM, 64 GiB RAM). Qwen3-30B-A3B is 17.3 GiB and
   fits **entirely in VRAM** — that is the demo, not V4-Flash. 96 GiB of fast
   memory against 144 GiB of model is ~67% resident there against ~11% here, so
   **measure it rather than predicting it**, and check `--auto` picks sensibly
   without the user knowing any flags.
3. **Verify the GPU tier.** `--device`, `-ngl`, `-ot`, `--op-offload` and
   `ggml_backend_sched` are all bound, on Vulkan — the sentence "no GPU code
   exists" was false for a day after it stopped being true. What is *not* done is
   verification: the device path fails 1 of 8 parity prompts where the CPU path
   fails none, and that is arithmetic rather than wiring.
4. Finish R5/T1-T5 of `lts-0-0-0.md`: quant selection, self-configuration.

**Dead ends, measured, do not re-propose**: expert factorisation, contextual
sparsity, a pinned hot set, expert-read/compute overlap (1.03x), `--op-offload`
(19% slower), `mul_mat_id` batching on the streaming path, and porting
parallel-experts to V4-Flash (its whole routed arithmetic is under 5% of a
token).

## Compact Instructions

If auto-compacted, preserve ONLY: open decisions, the work in progress, files modified this session, unresolved questions for Atur. Discard tool output, committed file contents, and dead ends.
