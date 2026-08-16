---
topic: the prompt for a session on a larger machine — run a huge model, wire it to a coding agent, PR the numbers back
status: ready to paste
links:
  - ../research/ram-frontier-qwen3-30b-2026-08-12.md
  - ../research/where-we-stand-vs-llamacpp-2026-08-16.md
  - ../reference/hard-won-facts.md
  - the-big-bang.md
---

# Prompt: Chaos on a bigger machine

**Everything between the two rules below is meant to be pasted into a fresh
Claude Code session on the other machine.** It assumes no context from this
repository or any prior conversation, which is the point — it has to work for a
stranger.

Precedent for keeping a prompt as a node: `worktree-prompt-tokenizers.md`.

**Why this session matters more than a demo.** The project's open question #1 is
*the tok/s-versus-resident-RAM frontier for a 144 GB model*. Nobody has published
it, and only an engine that owns residency can sweep it — `mmap` cannot be told
to use exactly N GiB. This laptop can only reach ~11% residency on V4-Flash, so
the curve measured here is all left-hand edge. A machine with more memory
measures the part that decides whether **20 tok/s is reachable on any consumer
hardware**, and at what price.

---

You are working on **Chaos**, an open-source Rust inference runner whose job is
running models that do **not** fit in RAM: it keeps the always-read weights
resident and streams the routed experts of a Mixture-of-Experts model from disk,
per token. Repository: <https://github.com/aturzone/Chaos> (Apache-2.0).

**This machine belongs to a friend of mine, not to me.** Be conservative. Do not
change system settings, do not uninstall or upgrade anything that was already
here, and do not write outside the directories we create together. Before any
download, tell me its size and where it will land. **If a step would use more
than 50 GB of disk, take more than an hour, or need administrator rights, stop
and ask me first.** When we are finished I want to be able to remove everything
we added by deleting two directories.

## What I want, in order

1. Find out what this machine actually is.
2. Get Chaos running on it.
3. Run the largest model it can hold, and measure it honestly.
4. **Sweep tok/s against resident RAM.** This is the one the project cares most
   about and the reason this machine is interesting.
5. Wire it to a coding agent, so I can write code against a local model.
6. Open a pull request to the project with what this machine measured.

Work through these in order and **stop after each numbered step to show me what
you found.** Do not run ahead.

## Step 0 — what is this machine

Install from a release rather than building, unless no archive matches this
platform:

- Download the archive for this OS from
  <https://github.com/aturzone/Chaos/releases> and unpack it.
- Windows: `powershell -ExecutionPolicy Bypass -File .\install.ps1`
- Linux/macOS: `sudo install -m 755 chaos-*/chaos-* /usr/local/bin/ && mkdir -p ~/.chaos/models`

Then run `chaos-probe` and report, as a table:

| | |
|---|---|
| total RAM / free RAM right now | |
| GPU model, VRAM total / free | |
| free disk on the fastest drive, and what kind of drive | |
| physical cores / logical cores | |
| OS and version | |

Then run `chaos-iobench <some-large-file>` — the archive ships it — and tell me
the **measured** scattered read bandwidth of that drive at 4 MiB slices, and
whether it improves past four concurrent handles. Do not skip this: on the
laptop this project was built on, the drive tops out at 2.74 GiB/s and stops
improving past four concurrent handles, and that single number sets the ceiling
for everything a streaming runner can do. **A quoted spec-sheet figure is not a
substitute.**

**Stop here and show me the table.** Which model we go for depends on it.

## Step 1 — the smallest thing that proves the install works

Get a small model — 2-4 GB, e.g. `Qwen3-4B-Q4_K_M.gguf` or
`Llama-3.2-1B-Instruct-Q4_K_M.gguf` — into the models directory the installer
created, then:

```
chaos-run                                    # should list what you just added
chaos-run qwen3 "The capital of France is" -n 16
```

**Two things that look like faults and are not:**

- **The first run that uses the GPU is slow, once.** ggml compiles its Vulkan
  shader set on first use and the driver caches it to disk afterwards, so run 1
  pays compilation *inside the timed region*. On the laptop that was 1.63 tok/s
  first and 9.0-9.6 tok/s every run after. **Never record a first run.**
- **Some architectures are refused by name and need `--force`.** An
  architecture only enters `VERIFIED_ARCHITECTURES` when someone has diffed its
  output against llama.cpp; `qwen3moe` and `deepseek4` are deliberately outside
  it. This is not a bug: **a wrong forward pass in this codebase produces fluent
  nonsense, never a crash**, so loading is not evidence and answering in English
  is not evidence.

## Step 2 — the largest model this machine can hold

Decide with me from the Step 0 table. Roughly:

| free RAM (+VRAM) | what is worth running |
|---|---|
| under 24 GiB | Qwen3-30B-A3B, 17.3 GiB — the standard MoE case |
| 24-80 GiB | Qwen3-30B fully resident, plus a 70B-class dense model |
| over 80 GiB | **DeepSeek-V4-Flash, UD-Q4_K_XL, ~144 GB across 5 shards** |

V4-Flash is the interesting one and it is a **144 GB download** — do not start it
without telling me the size, the destination and the time estimate, and confirm
there is at least 160 GB free. `chaos-model-info <file> --budget <GiB>` predicts
fit and tok/s **before** a download; use it and show me the prediction, so we can
compare it against what actually happens.

Once it runs, report the resident block Chaos prints — how much it wanted, how
much fit, what did not fit and what that costs per token.

## Step 3 — measure it, and follow this protocol exactly

The three published numbers this project has had to retract were all lost to
method, not to arithmetic. So:

- **Repeats, never one run.** Three at minimum, report the **median and the
  spread**. One sweep here gave `36: 72.41` against `99: 65.80` and an obvious
  causal story; three runs at the same point were 63.41 / 66.49 / 81.04, a spread
  wider than the whole effect being explained.
- **Alternate the arms inside one session.** Never compare a number you measured
  now against one written down earlier. This machine's state — page cache,
  thermals, whatever else is running — moves results by more than most effects
  worth measuring.
- **Discard the first run of anything that touches the GPU.**
- **Check for orphaned processes before trusting a surprising number.** A stopped
  benchmark wrapper once left a `llama-completion` alive holding 8.98 GiB, and
  every run after it read 10x slow and looked exactly like a regression.
  `Get-Process` / `ps` first.
- **A competitive claim is not citable until the competing command line and its
  output are pasted into the write-up.** If you compare against llama.cpp, run
  llama.cpp yourself, on this machine, alternating with Chaos, and paste both.
  Note that llama.cpp needs `--no-repack` to run a larger-than-RAM model — it
  *can* do it, and any claim to the contrary is false.

Measure, for each model: prefill tok/s (or ms per prompt token), generation
tok/s, and the resident footprint each was achieved at. **A tok/s number with no
footprint attached means nothing here.**

## Step 4 — the frontier, which is the actual deliverable

`chaos-run --cache <GiB>` sets the expert-cache budget explicitly. Sweep it
across the whole range this machine allows — e.g. 4, 8, 16, 32, 64, and as high
as free memory permits — with the **same prompt, same `-n`, same thread flags**,
three runs per point, interleaved rather than blocked (do a full pass over every
budget, then another pass, then a third — do not do three runs at 4 GiB and then
move on, or a machine that drifts will hand you a slope that is really a clock).

Produce a table with one row per budget: **budget, median tok/s, spread, GiB
actually streamed from disk, cache hit rate, free RAM at the start of the row.**

Free RAM per row is not decoration — an earlier attempt at this curve had a whole
round flattened by unrelated work releasing memory, and the only visible trace
was free RAM *rising* mid-round.

Then answer the two questions this is for:

1. **Where does the curve flatten, and why?** On Qwen3-30B it rose to 6 GiB and
   was flat after, for a capacity reason the engine reports directly: past that
   point `evictions` is 0 because the budget already covers what the run
   distinctly touches. If it flattens here, say which number proves it.
2. **What does the curve extrapolate to?** V4-Flash touches ~3.21 GiB of distinct
   expert weights per generated token. For 20 tok/s the bytes that come *from
   disk* have to fall to roughly 79 MB/token, which is about **97.5% of the
   expert bank resident**. Say what this machine's residency fraction actually
   is, what tok/s it actually gets, and whether the measured curve is consistent
   with that arithmetic — **including if it is not.** A measurement that
   contradicts the model is the most valuable thing this session can produce.

Include a **null control**: two budgets that are provably the same configuration
(both above the point where evictions hit zero) give you the noise floor for
free. Anything smaller than that spread is not a result.

## Step 5 — a coding agent against it

`chaos-serve <model> --port 8080` exposes an OpenAI-compatible endpoint. Wire up
**one** agent that accepts a custom base URL — `aider`, `Cline`, `Continue` or
`opencode` — pointed at `http://localhost:8080/v1`, and have it make one real
change to a small scratch repository so we know the loop closes.

Report: which agent, the exact configuration, tokens/second under a real coding
prompt (long context, not a one-liner), and **whether it is actually usable or
merely functional**. Say plainly if it is too slow to work with. Do not expose
the port beyond localhost.

## Step 6 — the pull request

Fork, branch `ticket/<name>`, and write up what you measured as a node in
`docs/graph/research/` named for the topic and dated. Read
`docs/graph/INDEX.md` first and add your node's line to it in the same commit —
that index is what every session reads first, so a node missing from it is a node
nobody finds.

Also read `docs/graph/reference/hard-won-facts.md` **before** proposing any
optimisation. About half of its entries are the measurement that killed an
appealing idea, and these in particular are closed with numbers: expert
factorisation, contextual sparsity, pinning a hot set, overlapping expert reads
with compute, `--op-offload`, `mul_mat_id` batching on the streaming path. Do not
re-propose them.

The write-up should carry raw numbers and command lines, not conclusions alone,
and should state its own caveats. **If this machine turns out to be slower than
expected, or if `--auto` chooses badly, that is the most useful possible result
and must go in exactly as measured.** Follow `CONTRIBUTING.md`.

Open the PR against `aturzone/Chaos` and give me the link.

---

## Notes for Atur, not part of the prompt

- The step order is a series of gates on purpose. The expensive, irreversible
  thing (a 144 GB download onto someone else's disk) sits behind two cheap steps
  and an explicit confirmation.
- Step 4 is the one to insist on if time runs short. Steps 2 and 5 are a demo;
  step 4 is a result nobody has published.
- If that machine is the 5090 box (32 GiB VRAM + 64 GiB RAM), note that
  Qwen3-30B-A3B at 17.3 GiB fits **entirely in VRAM** — that is the demo worth
  filming, not V4-Flash, which at 96 GiB of fast memory against 144 GiB of model
  is ~67% resident and still streaming.
