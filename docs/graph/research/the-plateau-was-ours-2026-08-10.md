---
topic: The 4-reader disk plateau was a synchronous-handle artefact, not the drive — and the expert matmul is 3% of a token, not the floor everyone assumed
status: measured, implemented
links: [v4flash-has-no-slack-2026-08-10.md, zero-copy-expert-reads.md, ../backlog/the-big-bang.md]
---

Two facts this project had written down were wrong, and both were wrong in the
same direction: they described a ceiling that was ours rather than the machine's.

## 1. Where a token actually goes

`v4flash-has-no-slack-2026-08-10.md` closed the byte-reduction roadmap and said
20 tok/s "needs the active weights to stop coming from disk". True, and it left
an unexamined assumption behind: that compute was ~1.0 s per token and therefore
a floor at ~1 tok/s.

`bigtea-kernelbench` (new) times the expert FFN with the weights **already in
memory**, so the arithmetic is measured without the disk in the way:

```
layer 20, 6 experts, 4096 x 2048 per matrix, 76.5 MiB resident
TOKENS   MS/PASS   MS/TOKEN   GFLOP/s     GiB/s   vs 1 TOKEN
     1      3.02       3.02     100.0     24.74        1.00x
     8      8.28       1.03     291.9      9.03        2.74x
    32     21.84       0.68     442.4      3.42        7.23x

reference  dense f32 1024x1024 matmul: 866 GFLOP/s on 20 threads
reference  single-threaded memcpy:     17.5 GiB/s (read+write)
```

**The expert matmul is 3.02 ms per block — 0.13 s per token across 43 blocks,
not 1.0 s.** And it is not inefficient: 24.7 GiB/s is *above* single-threaded
memcpy on this machine, which is what a matrix-vector product against 4-bit
weights should look like. The kernel is at DRAM bandwidth and there is nothing
to win there.

Measured against the real runner (`BIGTEA_BLOCK_TIMING=1`, single-token step,
2.66–3.11 GiB of the always-read set not resident):

| phase | before | share |
|---|---:|---:|
| dense always-read re-reads (disk) | 2.15 s | 39% |
| expert slice reads (disk) | 2.03 s | 37% |
| tail + graph overhead | 1.10 s | 20% |
| **expert matmul** | **0.18 s** | **3%** |

**76% of a token is disk.** The arithmetic everyone optimises for is 3%.

### The correction this forces

`v4flash-has-no-slack-2026-08-10.md` priced speculative decoding assuming verify
compute scales **linearly** with the batch, and concluded "wall-clock is worse
than the byte figure". That is wrong: measured compute scales as roughly
`n^0.49` — 8 tokens cost 2.74x one token, not 8x. Since expert *bytes* scale as
`n^0.667`, the whole pass is sublinear in both, and the byte table in that node
is a fair estimate of total speedup rather than an optimistic one. The
1.42x peak at α=0.9 stands; the extra pessimism does not.

## 2. The reader plateau was a Windows file handle

`CLAUDE.md` carried "four readers measured 1.59 → 1.99 GiB/s ... no further gain
at eight. The drive does 2.37 GiB/s sequential" — read as the drive's limit.

All four readers called `read_at` on **one shared `DirectFile`**. A Windows
handle opened without `FILE_FLAG_OVERLAPPED` is *synchronous*, and the I/O
manager serialises operations on it: concurrent `ReadFile` calls queue behind one
another however many threads issue them. The drive never left queue depth 1,
where an NVMe delivers a fraction of its rated throughput. The 1.59 → 1.99 gain
came from overlapping user-space work, not from the device.

`bigtea-iobench` (new) runs the identical scattered 4 MiB reads with one variable
— shared handle, or one handle per thread:

```
 THREADS    SHARED GiB/s      PER-HANDLE        GAIN
       1            1.54            1.61       1.04x
       2            2.08            2.32       1.12x
       4            2.01            2.65       1.32x
       8            2.05            2.69       1.31x
      16            2.06            2.60       1.26x
      32            2.06            2.62       1.27x
```

**Shared flattens at 2.05 GiB/s from two threads on. Per-handle climbs to 2.69**
— and 2.69 is also *above* the 2.37 GiB/s recorded as the drive's sequential
ceiling, so that ceiling was the handle too.

### Implemented

- `bigtea_model::Shard` opens a pool of `READER_HANDLES = 8` handles per shard,
  at load time rather than mid-stream. `Shard::reader(slot)` hands one out, and
  falls back to the primary handle if the pool could not be opened, so a
  descriptor limit costs throughput rather than the run.
- `Model::read_range_into_via(.., slot)` reads through a chosen handle.
  `read_range_into` is now a wrapper on slot 0, so no caller had to change.
- `READERS` in the deepseek4 path goes 4 → 8, one per handle. It must not exceed
  the pool or two readers collide on one handle again.
- `prefetch_dense` reads a block's **non-resident** always-read tensors across
  the pool before the bind loop. Binding cannot be parallelised — `ggml` contexts
  are not thread-safe and the graph must be built in order — but reading can, and
  that path was one tensor at a time through one handle.

### Measured end to end

Same prompt, same build except the change, `BIGTEA_BLOCK_TIMING=1`:

| | before | after | gain |
|---|---:|---:|---:|
| expert slice reads | 2.03 s | **1.54 s** | **1.32x** |
| dense re-reads, per GiB missing | 0.691 s | **0.496 s** | **1.39x** |
| single-token step | 5.46 s | 4.33 s | — |
| generation | 0.182 tok/s | 0.227 tok/s | — |

**The 1.32x on expert reads is the clean number** — it is independent of how much
happened to be resident, and it matches `bigtea-iobench`'s 1.31x prediction.

The step and tok/s rows are **not** a clean A/B: the two runs had 3.11 and 2.66
GiB of the always-read set missing respectively, so the second was slightly
favoured for a reason other than the change. Normalising the dense cost to equal
shortfall gives **1.19x** on the step, and that is the figure to quote. A clean
end-to-end A/B needs stable free RAM and has not been run.

## What this changes about the ceiling

At full residency a token is expert reads plus ~0.6 s of everything else. Expert
reads were 2.03 s and are now 1.54 s, and **`bigtea-iobench` says the drive has
no more to give at this access pattern** — 2.69 GiB/s is where per-handle
flattens too.

So the honest ceiling on this machine, with residency satisfied and reads
overlapped with compute (R2, not yet done), is roughly `max(1.54, 0.6)` ≈ 1.5 s
per token — **about 0.65 tok/s**, against llama.cpp's 0.39. That would be a
genuine 1.7x lead rather than parity, and every component of it is now measured.

It is not 20 tok/s, and nothing in this node changes that. What it changes is
that the remaining gap is **entirely disk bandwidth against 3.21 GiB per token**,
with compute at 3% and the kernel already at DRAM speed. Any further large win
has to come from bytes not travelling — which is residency — or from a device
with its own memory.

## What was NOT tested

- `FILE_FLAG_OVERLAPPED` with true asynchronous I/O and a completion port. The
  handle pool gets the same queue depth far more simply; overlapped I/O would
  only matter if the pool's thread-per-read model became the bottleneck, and at
  8 threads it is not.
- Linux and macOS. `O_DIRECT` handles are not serialised the same way, so the
  pool is expected to be neutral-to-positive there rather than a 1.3x. Untested —
  do not claim it.
- Whether `READER_HANDLES = 8` is right on a drive that is not this one. It is
  where *this* curve flattens; a self-configuring runner should measure it.

## Checked against the activation regression, and clear

**2026-08-12.** `ram-frontier-qwen3-30b-2026-08-12.md` established that a wrong
FFN activation changes *which experts get selected*, and therefore that a cache
measured on a model with the wrong one is measuring a different workload. Every
MoE residency figure in this repository was re-examined.

**This node is unaffected, twice over.** Everything measured here is
`deepseek4` / V4-Flash, which never tripped the bug — its first layers are dense,
so `blk.0.ffn_gate.weight` exists and the ungated-FFN detection saw a gate. And
the regression landed in `3573786` on **2026-08-11**, after this node was
written.

The 3%-of-a-token expert matmul figure and the per-handle reader numbers are
about bytes and syscalls rather than routing, so they would survive a routing
change regardless.
