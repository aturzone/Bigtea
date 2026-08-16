# The 6 GiB knee was a property of `-n 16`, not of the design

**2026-08-14.** Branch `ticket/r14-architectures`. Raw data:
[ram-frontier-surface-2026-08-14.csv](ram-frontier-surface-2026-08-14.csv).

The [frontier](ram-frontier-qwen3-30b-2026-08-12.md) was a *line* — tok/s against
cache size, swept at one generation length. This sweeps the second axis and the
answer is that the first curve described its `-n`, not the engine.

Links: [ram-frontier-qwen3-30b-2026-08-12.md](ram-frontier-qwen3-30b-2026-08-12.md) ·
[expert-cache-is-early-not-wrong-2026-08-08.md](expert-cache-is-early-not-wrong-2026-08-08.md) ·
[gpu-tier-smallest-honest-slice-2026-08-11.md](gpu-tier-smallest-honest-slice-2026-08-11.md)

## Read this before the numbers

**Qwen3-30B-A3B does not pass the parity diff** and is not in
`VERIFIED_ARCHITECTURES` — 0 FAIL but 6 of 8 prompts unstable, unexplained. The
sweep needs `--force` to run at all. It is the only container on this machine in
the size class where the curve is interesting, so this is published with the
caveat stated rather than withheld. Same standing as the node it extends.

## Method

```bash
# 3 rounds x -n {16,64,256} x --cache {1,2,4,6,8,12}, interleaved on BOTH axes,
# free RAM sampled before and after every row.
./target/release/chaos-run.exe -m Qwen3-30B-A3B-Q4_K_M.gguf \
    -p "The capital of France is" -n $N --temp 0 --cache $B --force
```

**Round 1 is discarded whole.** Free RAM before its rows ranged 0.30–12.19 GiB;
rounds 2 and 3 held 10.83–11.91 throughout. Round 1 reported 0.25 tok/s where
the clean rounds agree on 2.48 — a **10x error**, and the free-RAM column is the
only reason it is not in the table.

**Discard the round, not the row.** A naive "free ≥ 4 GiB" filter kept round 1's
`-n 16 --cache 6` row, which shows 7.45 GiB free and still ran 0.60 against a
clean 3.13. Free memory *at the moment before launch* does not tell you what the
machine did during the 40 seconds after it. Contamination is a property of the
period, so the round is the unit.

## The surface

tok/s is the mean of rounds 2 and 3. **`streamed`, `hit%` and `evictions` were
bit-identical in all three rounds** for every cell, contaminated ones included —
the workload is deterministic and only wall-clock moves.

| `-n` | cache | tok/s | spread | streamed | hit% | evictions |
|---|---|---|---|---|---|---|
| 16 | 1 | 1.72 | 2.3% | 12.13 | 35 | 1758 |
| 16 | 2 | 2.12 | 5.7% | 9.34 | 50 | 1957 |
| 16 | 4 | 2.48 | 0.8% | 6.69 | 64 | 1286 |
| 16 | **6** | **3.13** | 3.5% | **5.53** | 70 | **0** |
| 16 | 8 | 3.02 | 4.3% | 5.53 | 70 | 0 |
| 16 | 12 | 2.91 | 0.3% | 5.53 | 70 | 0 |
| 64 | 1 | 1.79 | 0.0% | 36.88 | 46 | 2756 |
| 64 | 2 | 2.28 | 2.6% | 24.33 | 64 | 3189 |
| 64 | 4 | 3.71 | 0.5% | 11.10 | 84 | 2285 |
| 64 | 6 | 4.24 | 9.4% | 7.57 | 89 | 557 |
| 64 | **8** | 3.68 | 1.6% | **7.05** | 90 | **0** |
| 64 | 12 | 4.38 | 7.1% | 7.05 | 90 | 0 |
| 256 | 1 | 1.58 | 0.0% | 146.72 | 45 | 3586 |
| 256 | 2 | 2.16 | 0.5% | 96.94 | 63 | 4697 |
| 256 | 4 | 2.94 | 0.7% | 41.07 | 84 | 3723 |
| 256 | 6 | 3.87 | 13.4% | 19.84 | 92 | 2675 |
| 256 | 8 | 4.18 | 13.6% | 12.48 | 95 | 1562 |
| 256 | **12** | **4.70** | 4.9% | **10.14** | 96 | **0** |

## The finding: the working set grows with what you generate

Read the `evictions = 0` rows. That is the budget at which the run stops
thrashing, and `streamed` there is the whole distinct working set:

| `-n` | working set | first budget with 0 evictions |
|---|---|---|
| 16 | 5.53 GiB | 6 |
| 64 | 7.05 GiB | 8 |
| 256 | 10.14 GiB | 12 |

**So "6 GiB is enough" was a statement about sixteen tokens.** At 256 it takes
12, and tok/s is still climbing at 12 rather than flattening — the knee for that
length is at or past the largest budget swept.

Growth is strongly sublinear: 16x the tokens for 1.83x the working set,
≈ `n^0.22`. Three points do not establish a law and the middle one sits 7% under
a two-point fit, so treat it as a shape, not a formula. The shape is enough for
the product question, and the answer is unwelcome: a 2048-token generation
extrapolates to **~14–18 GiB of expert cache**, and this machine has 15.7 GiB
total. At the 256-token knee the process already holds 12 GiB of cache plus 0.93
resident, leaving under 3 GiB for the operating system.

**The frontier is not a curve you can quote a single number off.** The honest
form of the product claim is *the largest model at the speed you want, for the
length you actually generate* — the third variable was doing real work and was
being held fixed at its most flattering value.

## More cache made it slower, at identical work

At `-n 16`, budgets 6, 8 and 12 read **the same 5.53 GiB, hit at the same 70%,
and evict nothing** — byte-for-byte the same work. They run 3.13, 3.02 and 2.91
tok/s. Same workload, more RAM, **7% slower**.

This is the [cache-hit-rate warning](../../../CLAUDE.md) measured under control
rather than inferred: past the working set, extra budget is memory the OS could
have used, and a cached byte that got paged out is a page fault wearing a hit's
disguise. The counter cannot see it, which is exactly why hit rate is not a
success metric. Note that it only appears where the budget *exceeds* the working
set — at `-n 256` nothing in the swept range is excessive and more is monotonic.

## What this does not settle

- **Where the 256-token curve turns over.** 12 GiB was the largest budget swept
  and it had not flattened. The next sweep needs 14 and 16, and 16 will not fit
  beside the OS on this machine.
- **Whether the exponent survives a second prompt.** One prompt, one model.
  Distinct-expert growth is a routing property and routing is per-prompt — that
  is [the R0 finding](routing-skew-is-per-prompt-2026-08-08.md), and it applies
  here.
- **The peak footprint of a run.** Free RAM is sampled before and after, so it
  catches a contaminated round but never sees the run's own high-water mark.
