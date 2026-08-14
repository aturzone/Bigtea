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

Two prompts produce a third answer. `Q: What is 17 plus 25? A:` is the one to
start on, because arithmetic has a right answer and prose does not:

```
llama.cpp default : 42\n\nOkay, so I need to figure out what 17 plus 25 is. Let me think.
llama.cpp -fa off : 42 A: 42\n\nOkay, let's
llama.cpp --no-repack : 17 + 25 = 42\n\nOkay,
bigtea            : \nOkay, let's see. The user is asking "What is 17 plus 25?" ...
```

Every reference configuration emits **42** before it starts reasoning, in three
different framings. Bigtea appears to go straight to the reasoning. If that holds
it is a real divergence and not a tie — the reference varies in *how* it states
the answer while always stating it, and we would be omitting it.

**This is not yet established.** The comparison above was captured with different
tail-truncation on the two sides and the confirming run was interrupted, so a
leading `42` on Bigtea's side cannot be ruled out from this evidence. The
measurement to run, at `-n 14` with identical post-processing on both sides, is
the first thing to do on this node. Do not cite it until then.

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
