# The tok/s-versus-offload frontier: a smooth dial, not a switch

**2026-08-16.** Qwen3-4B-Q4_K_M (2.33 GiB), RTX 3050 6 GB via Vulkan, i7-13650HX.
`bigtea-run -n 16 --temp 0`, three runs per point, **median** reported.

`-ngl` landed the same day with no performance number attached, which is a gap
worth closing rather than leaving: a placement flag whose effect on speed is
unmeasured is a flag nobody can use to make a decision.

| `-ngl` | blocks on card | prefill tok/s | generation tok/s |
|---:|---:|---:|---:|
| 0 | 0 / 36 | 43.29 | 6.34 |
| 9 | 9 / 36 | 48.38 | 6.41 |
| 18 | 18 / 36 | 54.57 | 6.99 |
| 27 | 27 / 36 | 63.78 | 7.06 |
| 36 | 36 / 36 | 66.49 | 7.78 |
| 99 | all + edges | **77.34** | **8.85** |

**Both curves are monotonic and there is no knee.** End to end that is 1.79x on
prefill and 1.40x on generation. The useful part is not the endpoint — it is
that every intermediate point is on the line, so `-ngl` is a dial a user can set
from how much VRAM they actually have, rather than an all-or-nothing switch.

The last step (36 → 99) moves the embedding and the output projection, and it is
worth more than any single block step: the vocabulary projection is the widest
matmul in the model.

## The single-run version of this table told a different story

Ten minutes earlier the same sweep, one run per point, produced:

```
36    prefill 72.41
99    prefill 65.80
```

— from which the obvious reading is "offloading the output head *costs*
something, prefill peaks at 36". That is false. The three runs at `-ngl 36` were
**63.41, 66.49, 81.04** — a 28% spread, wider than the entire 36→99 difference
being explained.

This is the third time this project has caught a confident causal story built on
one GPU run, and the first two both reached a published number. `bigtea-gpubench`
refuses `--repeat 1` without `--force` for exactly this reason; `bigtea-run` has
no such guard, so a sweep driven from the shell has to bring its own.

## What this is not

**The model fits.** 2.33 GiB against 5.11 GiB free, so every point on this curve
was a free choice rather than a constraint. The interesting frontier — *given N
GiB of VRAM, the largest model at the speed you want* — needs a model that does
**not** fit, and this is not that measurement.

It also is not a comparison. llama.cpp was not run on this ladder; the numbers
here are Bigtea against itself.

## Open

1. **The same sweep on a model larger than VRAM.** That is the curve CLAUDE.md
   names as the one nobody publishes, and `-ngl` is what makes it sweepable.
2. **Against llama.cpp at each point**, with the split matched on both sides as
   `parity-check.sh` now does for correctness.
