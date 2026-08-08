"""R0.1 — does a cache warmed on the prompt predict what generation needs?

    GGUF=... bigtea-run <model> "<prompt>" -n 16   # with BIGTEA_ROUTING_DUMP set
    python analyse_gen.py <dump.csv> [more.csv ...]

R0 established that the hot expert set is per-prompt: pinned from one prompt it
covers 53.7% of another, against 25.0% for caching at random. That leaves R1's
worth undecided, because R1 does not pin — it warms on the prompt it is actually
running. The question is what that warmed set is worth *afterwards*, on the
tokens the model goes on to generate. Between the two bounds R0 measured, the
answer is the difference between ~1.60 and ~7.76 tok/s.

**How one generated token's routing is isolated.** `bigtea-run` regenerates
statelessly: pass k re-runs prefill over prompt + k generated tokens. The model
is causal, so token i's routing is identical in every pass containing it — which
makes `pass[k] - pass[k-1]` exactly the routing of the single token generated in
between. The dump keeps passes apart so that subtraction is possible; the script
asserts the deltas are non-negative, which is the check that the causality
assumption held.

Two policies are scored, and the gap between them is the value of continuing to
warm the cache rather than freezing it after the prompt:

  frozen    top-k from the prompt alone, scored on every generated token
  warming   top-k from the prompt plus every token generated so far
"""

import os
import sys

import numpy as np

HASH_LAYERS = 3
N_EXPERT = 256
KS = (8, 16, 32, 64)


def load_passes(path):
    with open(path) as f:
        cols = f.readline().strip().split(",")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    if cols[0] != "pass":
        sys.exit(f"{path}: needs a per-pass dump (run a build with the `pass` column)")
    n_pass = int(raw[:, 0].max()) + 1
    n_layer = int(raw[:, 1].max()) + 1
    m = np.zeros((n_pass, n_layer, N_EXPERT), dtype=np.int64)
    m[raw[:, 0], raw[:, 1], raw[:, 2]] = raw[:, 3]
    return m[:, HASH_LAYERS:, :]


def topk_mask(counts, k):
    idx = np.argsort(-counts, axis=1, kind="stable")[:, :k]
    mask = np.zeros_like(counts, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def coverage(mask, counts):
    total = counts.sum()
    return float((counts * mask).sum() / total) if total else float("nan")


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)

    for path in paths:
        name = os.path.basename(path)[:-4]
        p = load_passes(path)
        if p.shape[0] < 2:
            print(f"{name}: only one pass — run with -n 16 to get generated tokens")
            continue

        deltas = p[1:] - p[:-1]
        neg = int((deltas < 0).sum())
        per_layer = deltas.sum(axis=2).mean()
        print(f"\n=== {name} — {p.shape[0]} passes, {len(deltas)} generated tokens ===")
        print(
            f"  causality check: {neg} negative cells"
            f" (must be 0), {per_layer:.1f} selections per layer per token"
        )
        if neg:
            print("  ABORT: a token's routing changed between passes; the delta is not one token")
            continue

        prompt = p[0]
        print(f"\n  {'top-K':>6} {'frozen':>9} {'warming':>9} {'in-prompt':>11} {'random':>8}")
        for k in KS:
            frozen_mask = topk_mask(prompt, k)
            frozen = coverage(frozen_mask, deltas.sum(axis=0))

            # Warming: before scoring token t, the cache has seen the prompt and
            # every earlier generated token. Scored token by token, then weighted
            # by selections so it is a true hit rate rather than a mean of means.
            seen = prompt.copy()
            hits = tot = 0
            for d in deltas:
                m = topk_mask(seen, k)
                hits += int((d * m).sum())
                tot += int(d.sum())
                seen = seen + d
            warming = hits / tot if tot else float("nan")

            print(
                f"  {k:>6} {frozen*100:8.1f}% {warming*100:8.1f}%"
                f" {coverage(frozen_mask, prompt)*100:10.1f}% {k/N_EXPERT*100:7.1f}%"
            )

        # Does it decay? If generation drifts away from the prompt, a frozen
        # cache should get steadily worse; if routing is stable within a
        # conversation it should not move.
        k = 64
        m = topk_mask(prompt, k)
        per_tok = [coverage(m, d) for d in deltas]
        first = np.mean(per_tok[: max(1, len(per_tok) // 3)])
        last = np.mean(per_tok[-max(1, len(per_tok) // 3) :])
        print(
            f"\n  drift (top-64, frozen): first third {first*100:.1f}%"
            f" -> last third {last*100:.1f}%"
        )
        print("  " + " ".join(f"{c*100:.0f}" for c in per_tok))

        # What the warmed hit rate is worth. Disk floor only: compute is extra,
        # and this project has already measured that a cached byte which gets
        # paged out is a page fault in disguise, so the cache must own its memory.
        gib_tok, drive = 3.21, 2.37
        print(f"\n  {'top-K':>6} {'cache':>10} {'warmed hit':>11} {'GiB/token':>10} {'disk floor':>12}")
        for k in KS:
            seen = prompt.copy()
            hits = tot = 0
            for d in deltas:
                m = topk_mask(seen, k)
                hits += int((d * m).sum())
                tot += int(d.sum())
                seen = seen + d
            hit = hits / tot
            miss = gib_tok * (1 - hit)
            print(
                f"  {k:>6} {137.06 * k / N_EXPERT:8.2f}G {hit*100:10.1f}%"
                f" {miss:9.3f} {drive/miss:9.2f} tok/s"
            )

    print("\nReference points from R0 (top-64): pinned across prompts 53.7%,")
    print("across subjects 37.5%, caching at random 25.0%, in-prompt oracle 90.5%.")
    print("llama.cpp generates this model at 0.21-0.31 tok/s on the same machine.")


if __name__ == "__main__":
    main()
