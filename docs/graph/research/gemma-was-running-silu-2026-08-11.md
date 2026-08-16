---
topic: Gemma-2 was listed as verified while running the wrong activation — and the second bug was one ULP
status: resolved, both Gemma-2 and Gemma-3 now byte-identical to llama.cpp
links: [gemma3-not-yet-2026-08-11.md, gemma2-sliding-window-2026-08-10.md, lts-parity-criteria.md]
---

Two bugs, found while chasing a third. `gemma2` was in
`VERIFIED_ARCHITECTURES`; it had never been diffed against llama.cpp.

```
$ chaos-run -m gemma-2-2b-it-Q4_K_M.gguf -p "The capital of France is" -n 16 --temp 0
 **Paris**.

$ llama-completion -m gemma-2-2b-it-Q4_K_M.gguf -p "The capital of France is" \
      -n 16 --temp 0 --no-warmup -no-cnv
:

a) Paris
b) Lyon
c) Marseille
d)
```

Both fluent, both English, neither the same model.

## Bug 1 — SiLU where the reference runs GELU

`grep -rn "gelu" crates/` returned nothing. Every gated feed-forward in the
crate was `down(silu(gate) * up)`. **The whole Gemma family uses GELU**, which
llama.cpp reaches through `LLM_FFN_GELU` → `ggml_geglu_split`:

```
$ llama-eval-callback -m gemma-2-2b-it-Q4_K_M.gguf -p "Hi" -n 1 --temp 0 -fa off
...
ffn_gate-0  = (f32)  MUL_MAT
ffn_up-0    = (f32)  MUL_MAT
ffn_geglu-0 = (f32)  GEGLU        <-- not SWIGLU
```

Nothing in a container records the activation: a GELU model and a SiLU model
hold **byte-identical tensor sets**. So the wrong choice is not a missing
tensor, not a shape mismatch, and not a crash. It is a model that keeps
answering, in a language you can read, wrongly. It is now `FfnAct`, chosen by
architecture name, applied in one place so a new FFN site cannot pick SiLU by
habit.

This alone fixed **Gemma-3**, which matched llama.cpp for 32 tokens on three
prompts immediately afterwards. It did not fix Gemma-2.

## Bug 2 — the same algebra, a different rounding

Gemma-2 differs from Gemma-3 in exactly one live feature: the attention logit
soft cap of 50. Disabling it made our output match llama.cpp exactly, which
made no sense — llama.cpp *does* apply the cap, and the eval-callback trace
proves it (`kq_scaled_1`, `kq_tanh`, `kq_scaled_2`).

The difference was where the `1/sqrt(head_dim)` goes.

| | Q entering the kernel | `scale` argument |
|---|---|---|
| llama.cpp | pre-scaled by 0.0625 | 1.0 |
| Chaos | raw | 0.0625 |

ggml folds the cap into the scale before the loop:

```c
if (logit_softcap != 0) { scale /= logit_softcap; }
...
s = s*scale;
if (logit_softcap != 0.0f) { s = logit_softcap*tanhf(s); }
```

so both compute `50·tanh(dot/800)` — in exact arithmetic. In f32 they do not:
`0.0625f/50f` rounds to `0x3AA3D70A` and `0.0625f*(1f/50f)` to `0x3AA3D709`.
**One ULP**, and it landed on a near-tie between `:` and ` Paris` as the first
token, which then rewrote the entire completion.

The lesson is not "floating point is hard". It is that **a soft cap turns a
scale into a non-linearity's argument**, and once a value goes through `tanh`
the last bit is no longer decorative. Anywhere the reference implementation
picks an order, match the order rather than the algebra. `prescale_q` records
which architectures llama.cpp pre-scales.

## Also fixed while in here: the 27B attention scale

llama.cpp picks the Gemma attention scale **by model size**:

```cpp
hparams.f_attention_scale = type == LLM_TYPE_27B
    ? 1.0f / std::sqrt(float(hparams.n_embd / hparams.n_head(0)))
    : 1.0f / std::sqrt(float(hparams.n_embd_head_k));
```

The two formulas coincide at every size *except* 27B — gemma-3-1b has
`head_dim` 256 and `n_embd/n_head` = 288, and its scale is `1/sqrt(256)`. So a
check that passed on the 1B would still have been wrong at 27B, on a model too
big to test here. `attn_scale_dim` encodes the rule rather than the
observation.

## Verification

Three prompts, 32 tokens, `--temp 0`, both engines, back to back. Output
identical token for token on **gemma-2-2b-it** and **gemma-3-1b-it**:

```
chaos-run -m <model> -p "<prompt>" -n 32 --temp 0
llama-completion -m <model> -p "<prompt>" -n 32 --temp 0 --no-warmup -no-cnv
```

| prompt | gemma-2-2b-it | gemma-3-1b-it |
|---|---|---|
| `The capital of France is` | identical | identical |
| `Once upon a time` | identical | identical |
| `def fibonacci(n):` | identical | identical |

llama, qwen2 and qwen3-4b re-checked and unchanged. `gemma3` added to
`VERIFIED_ARCHITECTURES`; `gemma2`'s entry is now earned rather than assumed.

## What this changes about the process

`VERIFIED_ARCHITECTURES` was supposed to be the defence against fluent
nonsense, and it contained an unverified entry for weeks. **Loading is not
evidence, and answering in English is not evidence.** The list now says so in
its own doc comment, and the diff is two commands.

The other outcome is `print_hparams`, at `-v`: llama.cpp has printed its
hyper-parameters at load since the beginning, and the hours spent guessing
which scale Gemma-2 was using were hours nobody with that output would have
spent. It prints *derived* values — `attn_scale`, the per-layer RoPE bases,
the windowed layer list — because a key read under the wrong name looks
exactly like a key that was absent until you print the result.

## The sweep the fix earned: `scripts/parity-check.sh`

Since the list had one entry nobody had checked, every entry got checked. Three
prompts, 32 tokens, `--temp 0`, both engines, seven containers:

| container | architecture | result |
|---|---|---|
| tinyllama-1.1b-chat | `llama` (SPM) | 3/3 identical |
| Llama-3.2-1B-Instruct | `llama` (BPE) | 2/3 identical, 1 unstable |
| Phi-3-mini-4k-instruct | `phi3` | 2/3 identical, 1 unstable |
| Qwen2-0.5B-Instruct | `qwen2` | 3/3 identical |
| Qwen3-4B | `qwen3` | 3/3 identical |
| gemma-2-2b-it | `gemma2` | 3/3 identical |
| gemma-3-1b-it | `gemma3` | 3/3 identical |

**19 of 21 exact, 0 failures, and the two exceptions are the interesting
part.**

### Greedy decoding is not stable under mathematical no-ops

`def fibonacci(n):` on Llama-3.2-1B:

```
$ llama-completion ... -fa on      -> "the Fibonacci sequence up to the nth term."
$ llama-completion ... -fa off     -> "the first n Fibonacci numbers."
```

`The capital of France is` on Phi-3-mini:

```
$ llama-completion ...             -> "Yes, that's correct. The capital of France is indeed Paris."
$ llama-completion ... --no-repack -> "Paris is known for its rich history, iconic landmarks such as..."
```

**The reference disagrees with itself**, under flags that only reorder a sum.
Those prompts sit on a near-tie: any implementation that accumulates in a
different order lands on the other side and then writes a different paragraph.
`-t` and `-b` do not move them, so this is not sloppiness — it is what a
near-tie looks like.

So "token-for-token identical to llama.cpp" is not always an achievable target,
and treating every mismatch as a bug would have sent someone hunting two that
do not exist. The script re-runs the reference under a second configuration
before calling anything a failure, and reports `unstable` instead. **A test
whose expected value is not reproducible in the reference must say so rather
than fail.**

This does not weaken the Gemma result: those were not near-ties. Gemma-2's
first token differed *and* the reference was stable, and after the fix the
match is exact on all three prompts.
