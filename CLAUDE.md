# Chaos — a runner for models larger than RAM

> **Read `STATUS.md` first.** It is the canonical statement of where the project
> is, the honest scoreboard, and what remains in order. Update it in the same
> commit as anything that moves a number or closes a task.

A Rust inference runner whose job is running models that do *not* fit in memory:
the always-read weights stay resident, routed experts stream from disk per token.
Borrows `ggml` for arithmetic; owns memory, residency, streaming and the token
loop. **Proven**: Qwen3-30B-A3B (17.28 GiB) generates correct text on a 15.7 GiB
machine, and DeepSeek-V4-Flash (144 GB) does too.

Graph docs live in `/docs/graph/`; read `INDEX.md` first, then only the 2–3 nodes
a task links to.

## Build / test / run

```
# ggml must be built first; point GGML_LIB_DIR at ggml-base.a, ggml-cpu.a, ggml.a
export GGML_LIB_DIR=C:/Projects/llamacpp-unsloth/build/ggml/src   # PowerShell: $env:GGML_LIB_DIR=...
# GPU work needs build-vulkan/ggml/src instead. That build has NO Vulkan archive,
# and the GPU tests SKIP rather than fail without a card -- so a green "6 passed"
# was once reported for a file whose two GPU tests never ran once.
cargo test --release          # 814 tests
cargo test --release --test deepseek4_forward -- --ignored   # 19 V4-Flash, needs the container
cargo build --release
./target/release/chaos-run <name-or-path> "prompt" -n 16   # bare `chaos-run` lists models
./target/release/chaos-probe --quick          # RAM/disk/GPU + what to close
./target/release/chaos-model-info <m.gguf> --budget 8   # fit + tok/s prediction
```

Windows needs the **GNU** Rust toolchain plus MSYS2 mingw64 on PATH, and
`.cargo/config.toml`'s `link-self-contained=no` must stay.

## Crates

`gguf` container parsing · `probe` hardware + RAM reclaim · `plan` prediction +
residency policy · `io` cache-bypassing aligned reads · `model` sharded
resolution, partial reads, name lookup · `ggml` FFI (graph, zero-copy weight
binding) · `tokenizer` byte-level BPE · `jinja` chat templates · `arch`
architectures + streaming forward pass · `image` PNG, safetensors, FLUX.2 VAE

## Traps — one line each, full text in `docs/graph/reference/hard-won-facts.md`

**Read that node before proposing any optimisation.** About half of its entries
are the measurement that killed an appealing idea.

- **ggml aborts rather than returning errors** — exhausted arena, bad device
  index, misaligned pointer. Size arenas up front, scale them with the prefill
  block, and check on the Rust side. It kills whole test binaries, not one test.
- **ggml `ne[0]` is the fastest dimension**; row-major reading transposes every
  matrix and yields confident nonsense. A PyTorch `[OC,IC,KH,KW]` conv weight is
  ggml `[KW,KH,IC,OC]` — the same bytes reversed, never transposed.
- **An image decoder is checked by round trip, never by looking.** Encode a real
  photo, score the reconstruction, and ablate the check first: three deliberate
  bugs each still produced a recognisable picture. `ggml_group_norm` applies no
  weight/bias, and `ggml_pad` is right/bottom-only — which is exactly what
  diffusers' downsampler wants.
- **`compute()` re-evaluates the whole ancestor graph** — call it only before a
  `to_vec_*`/`set_*`. **`compute(&t, 0)` runs on ONE thread**, not all cores.
- **Weights are bound zero-copy**; a copy needs 2× the model and will not fit.
- **A wrong forward pass produces fluent nonsense, never a crash.** Loading is
  not evidence, answering in English is not evidence — only a diff against
  llama.cpp counts, and that is what `VERIFIED_ARCHITECTURES` membership means.
  Nothing in a GGUF records the FFN activation; GELU-for-SiLU is silent.
  **An exit code is not a diff** — a sweep once reported twelve of twelve models
  working while one printed Thai. And **an architecture name is not a shape**:
  `qwen35` is exact at 24 blocks and nonsense at 64, so
  `catalogue::verified_block_counts` records what was diffed and both binaries
  warn otherwise.
- **When llama.cpp fails on the same file, the port is not what is wrong.**
  Chaos and llama.cpp agree to five significant figures on every layer of
  Qwen3.6-27B and then both go NaN at `l_out-5`; llama.cpp's own answer is
  `333333`. Run the competitor on the container before suspecting this engine.
- **Prompt length decides which code paths run** on V4-Flash — 2, 5, 165 and
  2048 tokens each reach a different attention builder.
- **Routing is not bitwise stable across sequence lengths**; a test demanding
  equal routing across batch shapes will fail on correct code.
- **Threads are two levers pulling opposite ways**: generation wants 2-4 (`-t`),
  prefill wants all of them (`-tb`). The MoE expert path wants **one**.
- **Cache hit rate is not a success metric** — a "hit" on bytes the OS has paged
  out is a page fault in disguise, and the tok/s-versus-budget curve plateaus
  (~6 GiB on Qwen3-30B) rather than climbing. Only tok/s at a stated footprint
  counts.
- **Score any residency policy out of sample**, with a uniform null and a noise
  ceiling. In-sample hot sets have lied here twice.
- **Only compare within one session.** V4-Flash drifts a lot with page-cache
  state, the first GPU run pays shader compilation inside the timed region
  (discard it), and an orphaned benchmark process holding 9 GiB looks exactly
  like a 10x regression — `Get-Process` before trusting a surprising number.
- **The load rate is not the re-read rate.** A load is one stream; a residency
  spill comes back across the 8-handle pool ~1.6x faster, and pricing the
  shortfall by the load rate oversold "close these apps" by half. A counter
  inside the R2-overlapped prefetch is worse — it measures occupancy (0.80
  GiB/s) not cost (2.44). Price it by re-reading the spilled tensors themselves.
- **The drive tops out at 2.74 GiB/s at four handles**, so the 8-handle pool is
  not the limit; the remaining gap is the per-block barrier, which cannot be
  filled because the next block's addresses depend on routing not yet computed.

## Working rules

- Git: remote `github.com/aturzone/Chaos`. Push with the token from
  `C:\Projects\.env` inline in the URL, output redacted — never in git config,
  never echoed, and **never `git push -u`**. Weights stay gitignored.
- Implementation goes on `ticket/<name>` branches + PR. **Claude owns git end to
  end**: merge when CI is green, close what it supersedes, delete the branch,
  prune, leave `main` verified. Docs may go to main.
- **Git hygiene, each rule bought with a mistake.** Verify containment with
  `git merge-base --is-ancestor <branch> origin/main` *before* deleting, never
  from "it was merged". After merging, `git checkout main` is not enough — a
  local `main` with no upstream makes `git pull` a silent no-op; fast-forward
  from `origin/main` explicitly and check a file only the merge added. Then
  **re-run tests on `main` itself**. GitHub parses only the *first* issue in
  `Closes #1, #2, #3`, so give every one its own `closes`.
- **A competitive claim is not citable until the competitor's exact command line
  and its output are in a doc**, from repeats, alternating in one session.
- Sync audit at phase boundaries only, not per commit.
- Keep this file under ~2000 tokens; tell Atur to prune rather than letting it
  bloat. It reached 3,308 words once — the overflow is now
  `reference/hard-won-facts.md`.

## Next

**v0.0.12 released 2026-08-20**: eight assets, five builds (Intel Mac and arm64
Linux are new). The window runs in the background — closing hides it to the
notification area, **Exit** is the only thing that stops the engine — updates
itself from inside the app, and `chaos-draw` ships as a binary.
`STATUS.md` is the scoreboard; `backlog/next-session-handoff.md` is the queue.

**Against llama.cpp, measured 2026-08-16 with both engines alternating**
(`where-we-stand-vs-llamacpp-2026-08-16.md`): **parity on everything that
streams** — V4-Flash prefill 1640 against 1679 ms/prompt token and generation
0.394 against 0.39, Qwen3-30B parity on both phases. Behind 1.20-1.27x on the
dense path when both sides are hand-tuned; ahead 1.23x out of the box, because we
measure the machine and llama.cpp uses a fixed default. **The old "V4-Flash
prefill 1.62x behind, generation 3-4x behind" is retracted**, and so is
"generation ~2x behind" on Qwen3-30B. Do not replace either with a claimed lead:
the ranges overlap.

**20 tok/s on V4-Flash is closed from both sides, with numbers**
(`v4flash-ram-frontier-2026-08-16.md`). Bytes: 20 tok/s needs 79 MB/token and it
reads 3288 (`v4flash-has-no-slack-2026-08-10.md`). Time: **a token is 1.56 s of
expert read plus 0.84 s that never touches the disk, so with EVERY expert
resident this CPU tops out at 1.19 tok/s** — the fixed cost alone is 17x over a
50 ms budget, and `-t` 2/4/8/16 confirms it is a floor, not a knob left wrong.
**The measured frontier**: 16 GB 0.42 (measured), 64 GB 0.55, 128 GB 0.93,
160 GB 1.19 — **holding the whole 144 GB model in RAM is worth 2.9x, not 48x.**
20 tok/s also needs 67.7 GB/s to the experts, so it is a GPU-memory
specification. Do not quote a GPU V4-Flash figure: resident-in-VRAM is untested
and the only measured number is 4.3x *slower* on streaming MoE.

1. **The frontier on a machine with real memory.** The curve above is this
   laptop's left-hand edge; the two numbers worth bringing back are `F` on a
   bigger CPU and `F` with the model resident on a real GPU, because the whole
   question reduces to them. Prompt ready: `backlog/bigger-machine-prompt.md`.
2. **A bigger machine**, measured rather than predicted. The 5090 box is 32 GiB
   VRAM + 64 GiB RAM, where Qwen3-30B-A3B (17.3 GiB) fits **entirely in VRAM** —
   that is the demo, not V4-Flash. 96 GiB of fast memory against 144 GiB of model
   is ~67% resident there against ~11% here. Check `--auto` picks sensibly
   without the user knowing any flags.
3. **Verify the GPU tier.** `--device`, `-ngl`, `-ot`, `--op-offload` and
   `ggml_backend_sched` are all bound on Vulkan; what is *not* done is
   verification — the device path fails 1 of 8 parity prompts where the CPU path
   fails none, which is arithmetic rather than wiring.
4. Finish R5/T1-T5 of `lts-0-0-0.md`: quant selection, self-configuration.

**Dead ends, measured, do not re-propose**: expert factorisation, contextual
sparsity, a pinned hot set, expert-read/compute overlap (1.03x), `--op-offload`
(19% slower), `mul_mat_id` batching on the streaming path, and porting
parallel-experts to V4-Flash (its whole routed arithmetic is under 5% of a
token).

## Compact Instructions

If auto-compacted, preserve ONLY: open decisions, the work in progress, files
modified this session, unresolved questions for Atur. Discard tool output,
committed file contents, and dead ends.
