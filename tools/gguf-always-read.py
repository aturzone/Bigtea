#!/usr/bin/env python3
"""Read a GGUF's tensor table over HTTP and report what must stay resident.

Adding a Mixture-of-Experts model to the catalogue needs two numbers: the
download size, which the Hugging Face API gives directly, and
`always_read_bytes` -- everything that is *not* a routed expert, which is the
figure that decides whether the model runs on a given machine at all. Getting
that wrong in the optimistic direction promises a user that a 22 GB model will
stream on a 16 GB laptop when it will not.

There is no API for it, so this reads the container's own header. A GGUF begins
with a tensor table naming every tensor, its shape and its type, and that is
enough to compute both totals exactly -- **without downloading the model**. The
table lives in the first few megabytes, so a ranged GET is all it costs.

A tensor is a routed expert if its name contains `_exps`, which is how llama.cpp
and this project both spell them (`blk.N.ffn_gate_exps.weight` and friends).
Everything else -- embeddings, attention, norms, the router, and any shared
expert -- is read on every token regardless of routing.

    python tools/gguf-always-read.py https://huggingface.co/<repo>/resolve/main/<file>.gguf

Prints the two numbers in the form the catalogue wants. Standard library only,
like everything else in `tools/`.
"""

import struct
import sys
import urllib.request

# Enough for the metadata block and the tensor table of a very large model.
# A 700-tensor container needs well under a megabyte; MoE files with thousands
# of expert tensors are the reason this is not 256 KB.
HEAD_BYTES = 24 * 1024 * 1024

# GGUF metadata value types.
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

# Block sizes and bytes-per-block for every quantisation llama.cpp defines, in
# ggml's own type order. `None` marks a type this script has never seen; hitting
# one is an error rather than a guess, because a wrong divisor here produces a
# confident and completely wrong size.
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    39: ("MXFP4", 32, 17),
}


class Reader:
    def __init__(self, buf):
        self.b = buf
        self.i = 0

    def take(self, n):
        if self.i + n > len(self.b):
            raise EOFError(
                f"the header is longer than the {len(self.b)} bytes fetched; "
                "raise HEAD_BYTES"
            )
        out = self.b[self.i : self.i + n]
        self.i += n
        return out

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        """Skip one metadata value of type `t`. Only the length matters here."""
        fixed = {U8: 1, I8: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32: 4, BOOL: 1,
                 U64: 8, I64: 8, F64: 8}
        if t in fixed:
            return self.take(fixed[t])
        if t == STRING:
            return self.string()
        if t == ARRAY:
            inner = self.u32()
            n = self.u64()
            for _ in range(n):
                self.value(inner)
            return None
        raise ValueError(f"unknown metadata type {t}")


def fetch_head(url, n=HEAD_BYTES):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status not in (200, 206):
            sys.exit(f"{url}: HTTP {r.status}")
        return r.read()


def tensor_bytes(dims, ttype):
    entry = GGML_TYPES.get(ttype)
    if entry is None:
        raise ValueError(f"unknown ggml type {ttype}; add it to GGML_TYPES")
    _, block, per_block = entry
    n = 1
    for d in dims:
        n *= d
    if n % block:
        # ggml pads, but a tensor whose fastest axis is not a multiple of the
        # block size means the assumption here is wrong, not the file.
        raise ValueError(f"{n} elements is not a multiple of block {block}")
    return n // block * per_block


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    url = sys.argv[1]
    buf = fetch_head(url)
    r = Reader(buf)

    if r.take(4) != b"GGUF":
        sys.exit("not a GGUF container")
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    print(f"gguf v{version}, {n_tensors} tensors, {n_kv} metadata keys",
          file=sys.stderr)

    # The architecture decides whether Chaos can run it at all: anything not in
    # `VERIFIED_ARCHITECTURES` has never been diffed against llama.cpp, and an
    # unverified forward pass produces fluent nonsense rather than an error.
    arch = "?"
    for _ in range(n_kv):
        key = r.string()
        v = r.value(r.u32())
        if key == "general.architecture":
            arch = v
    print(f"architecture      {arch:>18}")

    total = 0
    experts = 0
    expert_tensors = 0
    for _ in range(n_tensors):
        name = r.string()
        ndim = r.u32()
        dims = [r.u64() for _ in range(ndim)]
        ttype = r.u32()
        r.u64()  # offset
        size = tensor_bytes(dims, ttype)
        total += size
        # The routed experts, and only those, stream from disk per token.
        if "_exps" in name:
            experts += size
            expert_tensors += 1

    always = total - experts
    pct = always * 100 // max(total, 1)
    print(f"tensor bytes      {total:>18,}")
    print(f"routed experts    {experts:>18,}  ({expert_tensors} tensors)")
    print(f"always read       {always:>18,}  ({pct}% of the weights)")
    print()
    print("            always_read_bytes: {:_},".format(always).replace(",", ""))


if __name__ == "__main__":
    main()
