---
topic: D2 — GGUF v2 support, and the correction that the ticket's premise was about v1, not v2
status: resolved
links: [../backlog/lts-parity-criteria.md]
---

## The correction

D2 was written as:

> **D2 GGUF v2** — v2 writes array lengths as `u32` where v3 uses `u64`; build a
> v2 header in memory and test it.

**The `u32` to `u64` change was v1 to v2, not v2 to v3.** v2 and v3 have the
same field widths; v3's addition was big-endian support.

The evidence is llama.cpp's own reader, `ggml/src/gguf.cpp`. There is no width
branch anywhere in it — every length is read 64-bit — and the only version
handling is a refusal:

```cpp
if (ok && ctx->version == 1) {
    GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
    ok = false;
}
if (ok && ctx->version > GGUF_VERSION) { ... }
```

Had the ticket been implemented as written, the result would have been a `u32`
branch for v2 that reads real v2 containers wrongly — every string and array
length taken from the wrong four bytes — and a test built on the same wrong
premise would have agreed with it.

Chaos's reader already accepted `2..=3` with 64-bit lengths, so **it was already
correct**; what D2 actually needed was the proof, which is what this delivers.

## What is tested

`crates/chaos-gguf/tests/container_versions.rs`, with headers built byte by
byte in `tests/common/mod.rs` — no download, and the builder can declare a length
that disagrees with the bytes after it, which is what D1 needs next.

- **v2 and v3 parse identically.** Byte-identical payloads differing only in the
  version field produce equal metadata, equal tensor entries and the same data
  offset. That is the actual compatibility claim.
- A v2 container yields the values it declared — string, u32, string array, f32
  array, tensor shape.
- **v1 is refused**, matching llama.cpp.
- A future version is refused rather than guessed at.
- `general.alignment` is honoured when it is a power of two and ignored when it
  is not (0 and 33 both fall back to 32).
- A non-GGUF file is refused by magic.

## One thing added: naming an endianness mismatch

A v3 header written big-endian reads as `0x03000000` on a little-endian host.
The reader would have said *"unsupported GGUF version 50331648"* — true, and
useless: it reads like corruption and sends whoever hit it looking at the wrong
thing.

A version whose low 16 bits are zero is a small number with its bytes reversed,
so that is now its own error saying exactly that. llama.cpp makes the same check
for the same reason. The container is not corrupt; it is the wrong byte order for
the host, and those need different responses.

## Cost

7 container tests; 259 workspace tests pass, `clippy -D warnings` and `fmt`
clean. No model downloaded. The correction cost one `grep` of llama.cpp's
source, against an afternoon of implementing a `u32` branch that would have been
wrong and provable only against a real v2 container nobody has.
