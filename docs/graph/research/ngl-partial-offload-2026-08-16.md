# `-ngl` works, and it exposed that the device path was never diffed

**2026-08-16.** Llama-3.2-1B-Instruct-Q4_K_M, RTX 3050 6 GB via Vulkan,
`llamacpp-unsloth/build-vulkan`.

Partial offload runs. It is deterministic, it does not crash, and by the
project's own parity gate it is **exactly as accurate as full offload**.

What it turned up on the way is worth more than the flag. The device path fails
1 of 8 parity prompts where the CPU path fails none — a comparison nobody had
ever run — and chasing that produced the more useful result: **a greedy text
diff is not a valid acceptance test for a GPU path, in any engine.** The
kernels disagree by 0.37–0.71 in the logits while the model's own margin falls
to 0.399, so the flip is arithmetic. The device picks the same token on 8 of 8
first tokens. The measurement that shows this is in `chaos-gpubench` now,
because the old one — `sum(|logits[0..64]|)` to four decimals — could not have
failed.

Links: [phase-a-device-prefill-2026-08-15.md](phase-a-device-prefill-2026-08-15.md) ·
[mixed-residency-segfaults-2026-08-15.md](mixed-residency-segfaults-2026-08-15.md)

## The gate, with the split matched on both sides

`scripts/parity-check.sh` now takes `NGL=n` and passes `-ngl n` to **both**
engines. That is not a convenience; comparing our `-ngl 8` against llama.cpp's
`-ngl 0` would have measured the flag rather than the engine, because the
reference's own answer moves with the split (below).

```
export LLAMACPP_BIN=/c/Projects/llamacpp-unsloth/build-vulkan/bin
NGL=0  bash scripts/parity-check.sh Llama-3.2-1B-Instruct-Q4_K_M.gguf 32
NGL=8  bash scripts/parity-check.sh Llama-3.2-1B-Instruct-Q4_K_M.gguf 32
NGL=99 bash scripts/parity-check.sh Llama-3.2-1B-Instruct-Q4_K_M.gguf 32
```

| offload | ok | FAIL | other |
|---|---:|---:|---|
| `-ngl 0` — both on CPU | 6 | **0** | 1 unstable, 1 near-tie |
| `-ngl 8` — 8 of 16 blocks on the card | 7 | **1** | — |
| `-ngl 99` — everything on the card | 7 | **1** | — |

**Partial offload costs nothing over full offload.** Same score, and the two
failures are *different prompts* (`apples, oranges,` at 8; `def fibonacci(n):`
at 99), which is the signature of a near-tie landing differently rather than of
a broken split.

**The device path costs one prompt in eight.** `-ngl 0` fails none. That gap
belongs to `--device`, not to `-ngl`, and it has been there since Phase A —
which was accepted on "it runs and it is 1.73x", with no completion diff at all.

At this point in the day that read as an unclosed correctness bug. The logit
measurement below shows it is not; what it *is* is the absence of any check on
the device path until now.

## Greedy output moves with the split in llama.cpp too

The first reading of this data was wrong and worth recording. One prompt was
swept across `-ngl 0..17`; ours changed at 5 values, llama.cpp's did not change
at all, and the obvious conclusion was "our bug". Widening to eight prompts
reversed it:

```
llama.cpp, Llama-3.2-1B, --temp 0
  -ngl 0   A triangle has a base of 5 units and a height of 6 units.
  -ngl 99  A triangle has a base of 10 cm and a height of 6 cm.

  -ngl 0   The speed of light is approximately 186,000 miles per second.
  -ngl 99  The speed of light is approximately 299,792,458 meters per second.
```

Chaos flips on the same two prompts, and on `A triangle has` it flips the
*opposite* way. **A CPU kernel and a Vulkan kernel do not produce bit-identical
sums; greedy decoding turns a last-bit difference into a different word.** So
"the completion changed when I moved layers to the GPU" is not by itself
evidence of anything, in either engine — which is exactly why the gate has to
run both sides at the same `-ngl`.

The control that made the reference's numbers usable: `-ngl 99` prefills at
275.8 tok/s against 33.8 at `-ngl 0` on the same binary. Without that 8x, a
llama.cpp build that silently ignored `-ngl` would have looked perfectly stable
and proved nothing.

## Why `-ngl` needed no scheduler here, and where that stops being true

`ggml_backend_sched` landed the same day and **is not what makes this work.**
Worth stating plainly rather than letting the two changes borrow each other's
credit.

A mixed *graph* is undefined behaviour. A mixed *model* is not. This engine
materialises the activation as a host `Vec<f32>` at every block boundary — the
KV push takes bytes, the router reads a host vector, streamed expert bytes land
in host memory — so block 0's graph is wholly device-side, block 8's is wholly
host-side, and no single graph spans both. The per-block round trip is a cost
everywhere else in this project; here it is the thing that makes the split free.

When `backlog/activations-resident-across-layers.md` lands and `x` stops
returning to the host, that stops being true and the scheduler becomes load-
bearing. It is built and tested for that.

## One tensor broke the rule, and it segfaulted

`rope_freqs.weight`. Llama-3.1/3.2/3.3 ship their RoPE scaling as a tensor, it
carries **no `blk.` prefix**, and every block's rope consumes it. The first
residency rule read "no prefix means it is an edge tensor, host it unless
`-ngl` exceeds the block count", which put it in host memory while block 0 ran
on the card.

That is a mixed graph, and it did what this project has recorded twice before:
`Segmentation fault`, exit 139, **no error and no refusal**, immediately after
`TRACE: embedding ok`. Every `-ngl` from 1 to 16 died; 0 and 17+ were fine,
which is why a coarse sweep would have called it "partial offload is broken"
rather than naming the tensor.

It is bound twice now — once host, once device as `rope_freqs.weight@device` —
and the block picks by where it runs. Thirty-two floats. llama.cpp does not need
this because its single graph goes through the scheduler, which inserts the copy
itself.

**The general rule: a tensor every block reads must exist on both sides of a
split.** It is the only one in this engine today. A new architecture that adds
another will segfault exactly the same way.

## What was nearly published, twice

Both belong here because neither was caught by the harness.

**The GPU tests were skipping.** `CLAUDE.md` says
`GGML_LIB_DIR=.../llamacpp-unsloth/build/ggml/src`, and that build has **no
Vulkan archive**. `device_arithmetic.rs` and `scheduler.rs` both skip silently
without a discrete GPU, so `6 passed` was reported for a file whose two GPU
tests never ran. The scheduler commit's first draft claimed a mixed host/device
graph had computed. It had not. The Vulkan build is `build-vulkan/ggml/src`.

**The split assertion could not have failed.** `splits() >= 2` was asserted on
`mul_mat(host, device)` — a **single-node** graph. Splits partition nodes; a leaf
in another backend's buffer is copied in as an input and does not open a split.
The assertion was unfalsifiable, and it took a real GPU to reveal that, because
until then it was not being evaluated at all. The test builds two nodes now and
pins them to opposite backends.

## The 1-in-8 is arithmetic, not a bug — measured, same day

`chaos-gpubench` grew `--prompt <text>` and a real comparison. The old check was
`sum(|logits[0..64]|)` to four decimals, and **Phase A reported "logit checksums
agree" on it** — sixty-four entries of a 128k vocabulary, summed and rounded,
which cannot see the top token move. It is kept as a tripwire; the verdict below
is what answers the question.

All eight parity prompts, Llama-3.2-1B, CPU against device, first token:

| max abs Δlogit | CPU top-2 margin | prompt |
|---:|---:|---|
| 0.672 | 2.246 | The capital of France is |
| 0.420 | 2.421 | Once upon a time |
| **0.567** | **0.758** | def fibonacci(n): |
| 0.712 | 2.642 | 1 2 3 4 5 6 7 8 9 10 11 |
| 0.369 | 2.178 | The following is a list of items: apples, oranges, |
| 0.465 | 2.859 | Q: What is 17 plus 25? A: |
| 0.525 | 0.883 | SELECT name, COUNT(*) FROM users WHERE |
| **0.375** | **0.399** | Dear Sir or Madam, I am writing to |

**Every one picks the same token.** A wiring bug does not do that — it produces
nonsense, not agreement. What the table shows instead is the size of the two
quantities that decide the outcome: the kernels disagree by **0.37–0.71**
(mean 0.06–0.09), and the model's own margin falls as low as **0.399**. On
`Dear Sir or Madam` the difference is 94% of the margin. It agrees by a hair.

So within a 32-token greedy continuation, some position will have a margin under
0.4 and the token will flip. **That is arithmetic, and it is why a text diff is
not a valid acceptance test for a GPU path in any engine** — which is exactly
what llama.cpp's own 2-in-8 flip rate between `-ngl 0` and `-ngl 99` was already
saying.

The proper test is the one now in the tool: **argmax agreement, plus the
difference measured against the margin.** `parity-check.sh` cannot make that
distinction from text, and that is why the 1-in-8 was unresolvable there.

**What is still not established**: that 0.09 mean is *as good as llama.cpp's*.
Comparing the two engines' CPU-vs-GPU spreads needs llama.cpp's logits, which
has not been done. The claim here is narrower and survives: the device path
chooses the same token on 8 of 8, and the parity failures sit inside a band the
reference disagrees with itself in.

## Open

1. **Is our kernel spread larger than llama.cpp's?** Needs its logits, not its
   text. Until then "the device path is as accurate as the reference" is
   unproven in either direction.
2. **A speed number for `-ngl`.** Nothing here is a performance claim. The
   partial-offload tok/s ladder against resident VRAM is the interesting
   measurement and it has not been run.
