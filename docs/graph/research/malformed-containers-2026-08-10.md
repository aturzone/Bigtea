---
topic: D1 — a malformed GGUF must Err, never panic, and never quietly return the wrong value. Two sweeps that replace a fuzzing crate.
status: resolved
links: [../backlog/lts-parity-criteria.md, gguf-v2-premise-was-wrong-2026-08-10.md]
---

GGUF files are third-party input measured in hundreds of gigabytes, and every
length field in one drives an allocation. `Gguf::parse` takes `&[u8]` precisely
so a caller can hand it a few megabytes of a 144 GB file it has not validated —
which means a panic here is a denial of service in the server that embeds it.

Three properties, in the order they matter:

1. **No panics.** Whatever the bytes, `parse` returns.
2. **No unbounded allocation.** A declared count is a claim, not a fact.
3. **No wrong answers.** This one is the quiet one, and it is where the bug was.

## The bug: a duplicate key was a silent wrong value

Metadata was accumulated with `BTreeMap::insert`. A container holding
`general.architecture = "llama"` and then `general.architecture = "bert"`
**loaded as `bert`** — no error, no warning, no crash. A different reader taking
the first value would load the same file as a different model, and nothing in
either would say the file was ambiguous.

It is now `Error::DuplicateKey`. llama.cpp refuses these for the same reason, and
also refuses empty keys and duplicate tensor names; both of those are now
refused here too. Duplicate tensor names matter because names are how every
caller finds a tensor, so two with one name makes one of them unreachable and
which one arbitrary.

## The corpus

`crates/chaos-gguf/tests/malformed.rs`, headers built byte by byte with no
download. The builder can declare a length that disagrees with the bytes that
follow it, which is the case no real container provides and every hostile one
does.

| case | must be |
|---|---|
| string declaring more bytes than the file holds | `Truncated` |
| string declaring `u64::MAX` | `ImplausibleCount`, *before* allocating |
| array declaring more elements than follow | `Truncated` |
| array declaring `u64::MAX` | `ImplausibleCount` |
| unknown value type tag | `UnknownValueType` |
| array of arrays | refused, so the value parser stays non-recursive and cannot be stack-overflowed |
| **duplicate metadata key** | **`DuplicateKey`** |
| empty metadata key | `EmptyKey` |
| two tensors with one name | `DuplicateTensor` |
| tensor rank 99 | `ImplausibleCount` |
| `u64::MAX` tensor or metadata count | `ImplausibleCount` |
| non-UTF-8 string | `BadUtf8` |
| empty buffer | `Truncated` |

## The two sweeps

There is no fuzzing crate: the workspace has no external dependencies and that
is deliberate. Two exhaustive sweeps cover what a fuzzer would most likely find
here, and they need nothing:

- **Every prefix** of a valid container — truncation at every field boundary and
  every offset between them. Each must be an `Err`, and the full container must
  still parse, or the sweep proves nothing.
- **Every single-byte corruption** at five patch values across the whole header,
  >1,000 cases. Only that `parse` *returns* is asserted: most corruptions
  produce a container that is still structurally valid, and demanding an error
  would be demanding the parser detect corruption it cannot see.

A third targeted sweep replaces each declared length in turn with one that
overruns, because the byte sweep only reaches those by luck.

## What was already right

The reader was written defensively and most of this passed first time: bounds
checks on every read, `MAX_COUNT` and `MAX_STR` guards ahead of allocation,
`checked_add` on offsets, arrays-of-arrays rejected to keep the parser
non-recursive, and `with_capacity` clamped so a declared count cannot reserve
gigabytes. The value of D1 was the one case that was not defensive at all — the
one that returned an answer instead of failing.

## Cost

16 tests including two sweeps; 275 workspace tests pass, `clippy -D warnings`
and `fmt` clean. All three real containers on this machine still load
(`all-MiniLM-L6-v2`, `flan-t5-small`, `Qwen3-30B-A3B`), which is the check that
the new refusals do not reject files that are actually fine.
