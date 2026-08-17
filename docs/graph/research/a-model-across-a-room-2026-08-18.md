---
topic: Atur's idea — ten machines on one network running one model together
status: open, costed against measurement
links:
  - v4flash-ram-frontier-2026-08-16.md
  - v4flash-has-no-slack-2026-08-10.md
  - ../backlog/bigger-machine-prompt.md
---

# A model across a room

**Atur's proposal, 2026-08-18**: ten people, ten machines, one Wi-Fi network,
Chaos installed on each whatever the platform. They run one model together. One
person is chatting; the others are "token generators". A shared memory on one
machine accumulates what is produced.

**The verdict, costed against this project's own measurements: the memory half
is right and is the best idea anyone has had here. The token half cannot work,
for a reason that has nothing to do with engineering.**

## Why the memory half is right

The frontier node measured what V4-Flash costs per generated token on one
laptop: **1.56 s reading 3.15 GiB of expert weights off the disk, plus 0.84 s of
work that never touches the disk.** The disk read is 65% of a token, and it
exists *only* because 136 GB of expert bank does not fit in 16 GB of RAM.

Ten machines with 16 GB each is **160 GB of RAM in the room**. Minus what each
OS needs, roughly 120 GB usable — against a 136 GB expert bank. That is 88%
resident **across the group**, and the frontier table says a machine at that
residency runs about 0.9-1.0 tok/s against this laptop's 0.42.

Two ways to divide it, and they are not equivalent:

**Shard the experts.** Each machine holds a tenth of the bank in RAM. Per layer
six experts are needed and they land on up to six machines; each computes its
own and returns a 4096-wide vector. Traffic is about **8.3 MB per token**
(43 layers, an activation out and a result back for each expert).

| link | transfer | round trips | per token |
|---|---:|---:|---:|
| Wi-Fi 5, ~50 MB/s real | 0.17 s | 43 x 2 x ~3 ms = 0.26 s | **~0.42 s** |
| gigabit wired, ~110 MB/s | 0.08 s | 43 x 2 x ~0.3 ms = 0.03 s | **~0.10 s** |

Against **1.56 s of disk**. So the network replaces the bottleneck with
something 4x to 15x cheaper. Token time becomes ~1.26 s on Wi-Fi and ~0.94 s
wired: **1.9x and 2.5x** over today.

**Or split by layer** — machine 1 takes layers 0-4, machine 2 takes 5-9, and so
on. Then each machine holds ~14 GB of the model, dense weights and experts
together, **entirely in RAM with nothing streamed at all**, and only the
activation moves down the chain: ten hops of 16 KB, about **160 kB per token**,
fifty times less traffic than sharding. This is the better arrangement, and it
is the one to build.

Either way the conclusion is the same and it matters: **a model too large for
any single machine in the room becomes a model that runs at memory speed on the
group.** That is this project's whole thesis — owned residency — extended from
one box to several, and it reaches the right-hand side of the frontier curve
without anyone buying a 160 GB machine.

## Why the token half cannot work

The proposal has the other nine machines generating tokens in parallel and
pooling them. **Token N+1 cannot start before token N exists**, because it is
computed *from* token N: the model reads everything written so far to decide the
next word. Nine machines cannot produce nine consecutive words at once any more
than nine people can dig a hole to nine times the depth by each digging their
own.

This is not an implementation difficulty. It is what autoregressive generation
means, and no engineering removes it.

**What the nine can do instead, all real:**

- **Hold the weights** — the memory half above. The big one.
- **Serve nine other people at once.** With layers split across the group, while
  machine 2 works on your token, machine 1 can already be working on someone
  else's. Total output across the room scales close to linearly with machines.
  **Throughput, not latency.** Ten conversations at 1 tok/s each, not one at 10.
- **Guess ahead.** One machine runs a small fast model proposing several words;
  the group checks them all in one pass and keeps the correct prefix. Measured
  here at **~1.4x, not the 2.2x the literature claims**, because a verification
  pass over more tokens touches more distinct experts
  (`U(n) ≈ 6·n^0.667`) — and below ~75% acceptance it is a net loss.

Atur's instinct about "centralised memory on one system" is right and should
stay: the conversation and its KV cache belong on the machine of the person
chatting. Everyone else is stateless, which also makes a machine leaving the
room a non-event.

## So what would the room actually get

For **one person chatting**, with layers split ten ways and everything resident:

```
  disk read      1.56 s  ->  ~0.02 s   (nothing streams; 160 kB crosses the wire)
  fixed work     0.84 s  ->   0.84 s   (unchanged -- it is 43 layers in sequence)
  ------------------------------------
  token          2.40 s  ->  ~0.86 s   =  ~1.2 tok/s
```

**About 2.8x, and then it stops.** The floor is the same 0.84 s the frontier node
found, for the same reason: a token must pass through all 43 layers in order, and
splitting them across machines changes where the work happens, not how much of it
is sequential. Ten machines land in the same place as one machine with 160 GB of
RAM — which is exactly what ten machines *are*.

**This does not reach 20 tok/s and nothing on the CPU path does.** It does reach
the top of the curve for free, out of hardware that is already in the room.

## Worth knowing before building it

- **Wi-Fi latency, not bandwidth, is the risk.** 43 layers is 86 round trips per
  token; at 3 ms that is 0.26 s before a single byte of payload. On a congested
  network with ten peers it is worse, and it is the number to measure first.
  Layer-splitting is chosen partly because it makes 10 hops instead of 86.
- **Ten machines must agree bit-for-bit.** Routing here is already not stable
  across sequence lengths, and a peer running a different build would diverge
  quietly — as fluent nonsense, never as an error, which is this codebase's
  signature failure. Every peer must report its version, and a mismatch must
  refuse rather than proceed.
- **A peer that leaves mid-token takes the answer with it** unless its layers are
  held somewhere else too. Redundancy costs memory, which is the thing being
  saved.
- **Trust.** Anything on that network can be handed your prompt. Fine for ten
  friends in a room, not fine as a default.

## Next

Not scoped as tickets yet. The first thing worth doing is the cheapest and does
not need ten machines: **measure round-trip latency and throughput between two
of them**, and check the 86-round-trip and 10-hop estimates above against a real
network. If the latency lands near 3 ms, layer-splitting is viable and expert
sharding is not.
