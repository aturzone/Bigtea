---
topic: Gemma-2's sliding-window attention, implemented and verified above the window against llama.cpp — the 4096-token refusal is gone. Three arenas were short, and the one that mattered was found by reading ggml's error correctly
status: measured
links: [lts-parity-criteria.md, threads-were-never-plumbed-2026-08-10.md, qwen3-4b-vs-llamacpp-2026-08-10.md]
---

Gemma-2 alternates a **sliding-window** layer with a **full-attention** one.
Bigtea implemented neither the window nor a way to live without it, so
`correct_context_limit()` refused any sequence past 4096 tokens. That refusal
was honest — below the window every layer *is* effectively full attention, so
short sequences were exactly right — but it made the only Gemma container on
this machine unusable at the lengths anyone cares about.

## What was added

A second mask. The causal mask is built once per pass and shared by every layer;
the windowed layers now get a copy with the **old** keys closed off as well:

```rust
let oldest_visible = absolute - window + 1;
for key in 0..oldest_visible.min(n_total_final).max(0) {
    m[at..at + 2].copy_from_slice(&F16_NEG_INF);
}
```

and the layer picks between them:

```rust
let layer_mask = match swa_mask.as_ref() {
    Some(swa) if il % 2 == 0 => swa,   // even layers slide
    _ => &mask,
};
```

## Verification: three checks, because two of them prove nothing alone

**1. Below the window, output must not change.** `"The capital of France is"`
still gives `**Paris**.` — the two masks are identical there, so this is a
regression check and nothing more.

**2. Above the window, output must match llama.cpp.** 5201 tokens, greedy, raw
completion on both sides:

```
$ bigtea-run gemma-2-2b-it-Q4_K_M.gguf -f target/p5k.txt -n 16 -t 4
prefill 5201 tokens in 45.2s (114.99 tok/s)
"The history of computing is a history of abstraction. Each generation of engineers built a"

$ llama-completion -m gemma-2-2b-it-Q4_K_M.gguf -f target/p5k.txt -n 16 -t 4 \
    -c 8192 --temp 0 --no-warmup -no-cnv
"The history of computing is a history of abstraction. Each generation of engineers built"
```

**`-no-cnv` matters.** Without it `llama-completion` applies Gemma's chat
template, answers as an assistant (*"This is a fascinating and insightful
observation!"*), and the two engines are not doing the same thing at all. The
first comparison run made exactly that mistake.

**3. The layer parity must be load-bearing.** Check 2 used a repetitive prompt,
and a repetitive prompt continues itself under almost any attention — so
matching llama.cpp there could be luck. Flipping the parity to `il % 2 == 1`
and re-running the *same* prompt gives:

```
"The history of computing"
```

Different output. **The mask is doing work at this length, and only the even-slide
ordering reproduces llama.cpp.** Without this third check the first two are
consistent with the window never being applied.

## Three arenas were short, and only one of them was the cause

`ggml` does not return an error when an arena is exhausted — it calls
`GGML_ASSERT` and the process dies. The abort read:

```
ggml_new_object: not enough space in the context's memory pool
  (needed 56624208, available 54532608)
```

**`available` is the pool's total size, not what is left in it.** Reading it as
the remainder sends you to whichever arena happened to be nearly full instead of
the one that was too small — which is what happened here, and cost two wrong
fixes before the arithmetic was done:

`56,624,208 ≈ 3 × 18,874,368` = three `n_embd × n_new` f32 tensors at
`n_embd=2304, n_new=2048`, plus object headers. And
`54,532,608 − 37,748,736 (= 2 × 18,874,368) ≈ 16 MB` is the graph reserve. So
the failing arena budgeted **one** such tensor and allocated **three**.

It was `post_norm` — Gemma's alone, which is why no other architecture ever hit
it:

| arena | budgeted | actually allocates |
|---|---|---|
| `post_norm` | 1 × `n_embd × n_new` | **3** — input, rms intermediate, scaled result |
| dense FFN | `n_ff × 4`, `n_embd × 5` | + 2 more `n_embd` for Gemma's post-FFN norm |
| attention | q-cont, k-cont, v-cont, … | + the un-permuted q, and k and v read from the cache |

Only the first caused this abort. The other two were genuinely under-counted and
are fixed here as well — they would have aborted at some larger block, which is
the same bug arriving later. **`arena_for` doubles its total, and that doubling
is what hides an undercount until the block grows enough to eat it.** Every
tensor a branch can allocate has to be listed for that branch.

## Performance, and a number that looked like a win and was not

At `-t 4` on both sides, Bigtea prefilled 5201 tokens at **114.99 tok/s** against
llama.cpp's **76.76** — 1.50x ahead. That figure is worthless: prefill is
compute-bound and wants every core, so `-t 4` handicaps it, and llama.cpp was
being run at a setting nobody would use.

```
$ llama-completion ... -t 20    prompt eval 127.35 tok/s / 5200 tokens
$ bigtea-run       ... -t 20    prefill     109.30 tok/s / 5201 tokens
$ bigtea-run       ... -t 4     prefill     114.99 tok/s / 5201 tokens
```

| Gemma-2-2b prefill, 5200 tokens | best of each | verdict |
|---|---:|---|
| llama.cpp | **127.35** (t=20) | — |
| Bigtea | 114.99 (t=4) | **1.11x behind** |

**Not a win.** Recorded because the 1.50x version was one command line away from
being quoted, and this project has retracted two claims already. The rule that
caught it is the one already in `CLAUDE.md`: run the opposing command, at the
setting its own author would choose.

## What this does not cover

- **Gemma-3**, which uses a 5:1 window pattern rather than 1:1, and a different
  window size per layer group. Not implemented, and the parity constant here is
  hardcoded to Gemma-2's alternation rather than read from metadata.
- **Memory.** The window means old KV entries can never be read by half the
  layers, so their cache could be a ring of `sliding_window` entries instead of
  the full sequence. Bigtea still stores all of it — 529.8 MiB at 5216
  positions. That is a real saving left on the table, and it is the same
  wraparound work as issue #46.
- **Sequences past ~8k**, untested; the arenas now scale but nothing above 5201
  tokens has been run.
