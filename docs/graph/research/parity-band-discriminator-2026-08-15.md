---
topic: "`unstable` was answering the wrong question — four of qwen3moe's six near-ties reproduce llama.cpp's own output byte for byte, and two do not"
status: measured 2026-08-15, two prompts still open
links: [llamacpp-flag-audit.md, routing-skew-is-per-prompt-2026-08-08.md]
---

## The wrong question

`scripts/parity-check.sh` classified a disagreement by asking:

> does llama.cpp disagree with **itself** here, under flags that only reorder a sum?

and reported the answer as though it settled a different question:

> is Bigtea's output **one of the things** llama.cpp disagrees between?

Those come apart exactly where it matters. A prompt where the reference wobbles
between A and B, and Bigtea says B, is a tie — our answer is the reference's own,
arrived at by a different route. A prompt where the reference wobbles between A
and B and Bigtea says C is not explained by that wobble at all. **The old harness
printed the same word for both**, and the nine-of-eleven `unstable` verdicts that
turned out to be real bugs were all the second kind.

## What it changes on qwen3moe

Same model, same eight prompts, same build. Old harness, and the r14 session's
independent run agrees with it:

```
2 ok, 6 unstable, exit 2      -> "three-in-eight fires; there is a real bug"
```

With the discriminator:

```
ok        def fibonacci(n):
ok        1 2 3 4 5 6 7 8 9 10 11
near-tie  The capital of France is                    reproduces `-b 1`        exactly
near-tie  Once upon a time                            reproduces `-fa off`     exactly
near-tie  The following is a list of items: ...       reproduces `-b 1`        exactly
near-tie  SELECT name, COUNT(*) FROM users WHERE      reproduces `-b 1 -fa off` exactly
unstable  Q: What is 17 plus 25? A:                   matches none of them
unstable  Dear Sir or Madam, I am writing to          matches none of them

4 of 8 in-band, 2 outside, exit 0
```

**Four of the six were never unexplained.** On those, Bigtea's continuation is
byte-identical to something llama.cpp itself emits under a configuration that
only reorders a sum. That is the strongest evidence available short of matching
the default, and the old harness could not express it.

The evidence for a bug in the qwen3moe path is therefore **2 of 8, not 6 of 8**.
It is not zero, and the two are below the cluster threshold rather than absent.

## The variation is itself the evidence

Which configuration we land on is **not constant**: `-b 1` twice, `-fa off` once,
`-b 1 -fa off` once. That distinction is worth more than the count.

A systematic defect would be systematic. If Bigtea were, say, quietly running
batch-1 semantics on a batched prefill, it would reproduce `-b 1` on *every*
prompt where the default differs — a single wrong behaviour reproducing a single
reference configuration. Landing on three different configurations across four
prompts is what a genuine near-tie looks like: the tie breaks whichever way the
rounding happens to fall, and the reference's own configurations are scattered
the same way.

**So the discriminator is a diagnostic, not just a verdict.** *Which* member of
the band we match names the behaviour we share, and a constant answer would be a
lead. This one is not constant.

## The composed configuration earned its place immediately

`-b 1 -fa off` was added on the r14 session's Phi-3 measurement — neither flag
alone reproduced Bigtea's answer there, only the two together. On its first run
against a different model it explained the `SELECT` prompt, which no single flag
accounted for. **A near-tie that needs two no-ops composed is invisible to a
probe that tries flags singly**, and that class is not rare; it is the one nobody
had looked for.

## What is still open

Two prompts produce a third answer. `Q: What is 17 plus 25? A:` was the one to
start on, because arithmetic has a right answer and prose does not.

**Measured, `-n 14`, identical post-processing on both sides:**

```
default        ' 42\n\nOkay, so I need to figure out what '
-b 1           ' 42\n\nOkay, so I need to figure out what '
-b 1 -fa off   ' 42\n\nOkay, so I need to figure out what '
-fa off        " 42\nA: 42\n\nOkay, let's"
--no-repack    ' 17 + 25 = 42\n\nOkay,'
BIGTEA         "\n 42\n\nOkay, let's see. The user is asking"
```

**Bigtea emits `42`.** The earlier reading — that it skipped the answer and went
straight to reasoning — was **wrong, and was an artefact of the measurement**:
the two sides had been captured with different tail-truncation, so Bigtea's first
line was cut off and the reference's was not. This node flagged that claim as not
citable before anyone acted on it, which is the only reason it cost nothing.

What the corrected data shows:

* **Every engine and every configuration answers 42.** On the single
  outside-band prompt that has a checkable answer, our answer is correct and
  identical to the reference's. The disagreement is entirely in the prose after
  it.
* **The reference spans three distinct outputs across five configurations here**
  — including `A: 42` repeated on its own line under `-fa off`, and
  `17 + 25 = 42` under `--no-repack`. This prompt sits in a high-entropy region
  *after* the answer, where the continuation is barely determined at all.
* Bigtea is a fourth output, so `outside the band` is correct as classified. But
  it agrees with `-fa off` on the token where the reference splits (`Okay,
  let's` against `Okay, so`) and differs from it only by the repeated `A: 42`
  line.

**This is now weak evidence of a defect, not strong.** A wrong forward pass that
survives to token 14 with the arithmetic intact, on a prompt where the reference
cannot agree with itself three ways, is not the shape of the bugs this harness
has caught before — Llama-3.2's RoPE and Falcon3's short prefill both broke
prompts that had a determined answer.

The second outside-band prompt, `Dear Sir or Madam, I am writing to`, is open
prose with no checkable answer and is therefore the weaker of the two to reason
from. It has not been examined token by token.

## Every architecture, re-scored under it

All thirteen dense architectures, eight prompts each, `-n 32`, `--temp 0`,
against `llama-completion`:

| model | exact | near-tie | outside | exit |
|---|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | 8 | 0 | 0 | 0 |
| tinyllama-1.1b-chat | 8 | 0 | 0 | 0 |
| Qwen2-0.5B-Instruct | 8 | 0 | 0 | 0 |
| Qwen3-4B | 8 | 0 | 0 | 0 |
| **Phi-3-mini-4k** | 6 | **2** | 0 | 0 |
| Falcon3-1B-Instruct | 8 | 0 | 0 | 0 |
| gemma-2-2b-it | 8 | 0 | 0 | 0 |
| gemma-3-1b-it | 8 | 0 | 0 | 0 |
| stablelm-2-1_6b-chat | 8 | 0 | 0 | 0 |
| starcoder2-3b | 8 | 0 | 0 | 0 |
| OLMo-1B | 8 | 0 | 0 | 0 |
| internlm2-math-plus-1_8b | 8 | 0 | 0 | 0 |
| baichuan2-7b-chat | 8 | 0 | 0 | 0 |
| **total** | **102** | **2** | **0** | 13/13 |

**Nothing lands outside the band on any of the thirteen.** Both near-ties are
Phi-3, and one of them reproduces llama.cpp's `-b 1 -fa off` output — the
composed configuration, on a model it was derived from, confirming that the class
is real rather than an artefact of the model it was found on.

`qwen3moe` is the fourteenth and is scored above: 2 exact, 4 near-tie, 2 outside.
It is the only model with anything outside the band, which is itself worth
noting — whatever is happening there is specific to it and not a property of the
harness or of the engine in general.

**This is evidence about these eight prompts on these thirteen models**, and
nothing more. `starcoder2` passed 3/3 once while running the wrong pre-tokenizer.
V4-Flash is not swept here.

## The threshold moved without moving

Three-in-eight still fails, but it now counts the sharper class, and that is a
change in stringency that should be said out loud: the threshold was calibrated
on a population where both situations shared a word. Two outside-band answers
under the new rule is stronger evidence of a defect than two `unstable` was under
the old, because everything excusable has been taken out of the class.

A bound was added in the other direction for the same reason. Every configuration
added to the probe widens the band, so "in band" gets cheaper as the probe grows
— the direction that turns a harness into a rubber stamp. Six ties in eight now
fails as well: each one individually explained is still not what a matching
engine looks like.
