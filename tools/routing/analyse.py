"""R0 — is V4-Flash's hot expert set global, or per-prompt?

    ./capture.sh && python analyse.py [captures/csv]


v0.0.2 measured one coding prompt and published "top-64 of 256 experts absorb
97.8% of selections", which is the premise under the whole hot-set cache plan.
Two things were never checked, and both can only inflate that figure:

1. **Sample size.** A prompt of N tokens makes 6N routing decisions per layer.
   If 6N is not >> 256, the top-64 cover most of the mass *by construction* —
   you cannot spread 102 draws over more than 102 experts. Corrected here by
   Monte-Carlo: the same number of draws from a *uniform* router, which is the
   null the original compared against.

2. **In-sample scoring.** The hot set was chosen from the same prompt it was
   scored on. A cache is pinned before the prompt arrives, so the honest metric
   is coverage of prompt B by a hot set built from prompt A. That number is
   unbiased and needs no null correction — it *is* the cache hit rate.

Verdict rule: cross-prompt coverage near the in-sample figure means the hot set
is global and can be pinned. Near the random floor (K/256) means it is
prompt-dependent and must be warmed adaptively, which is a different design.
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "captures", "csv")

# blk.0-2 select out of ffn_gate_tid2eid by token id, not by a learned gate.
# Their "skew" is the token distribution, so they say nothing about routing.
HASH_LAYERS = 3
N_EXPERT = 256
KS = (8, 16, 32, 64, 128)
RNG = np.random.default_rng(20260808)


def load_passes(path):
    """[pass, layer, expert] counts, learned-gating layers only.

    Accepts both dump formats: `pass,layer,expert,count` from the current
    binary, and the original `layer,expert,count`, which is read as one pass.
    """
    with open(path) as f:
        cols = f.readline().strip().split(",")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    if cols[0] != "pass":
        raw = np.column_stack([np.zeros(len(raw), dtype=np.int64), raw])
    n_pass = int(raw[:, 0].max()) + 1
    n_layer = int(raw[:, 1].max()) + 1
    m = np.zeros((n_pass, n_layer, N_EXPERT), dtype=np.int64)
    m[raw[:, 0], raw[:, 1], raw[:, 2]] = raw[:, 3]
    return m[:, HASH_LAYERS:, :]


def load(path):
    """[layer, expert] counts pooled over passes, learned-gating layers only.

    Pooling is right for *shares* — a repeated pass scales every bin alike — and
    wrong for chi-square. Capture with `-n 1` for anything distributional.
    """
    return load_passes(path).sum(axis=0)


def topk_mask(counts, k):
    """Per layer, a boolean mask of that layer's k most-selected experts."""
    idx = np.argsort(-counts, axis=1, kind="stable")[:, :k]
    mask = np.zeros_like(counts, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def coverage(mask, counts):
    """Share of `counts` selections that land inside `mask`. The hit rate."""
    total = counts.sum()
    return float((counts * mask).sum() / total) if total else float("nan")


def noise_ceiling(counts, k, trials=60):
    """The coverage two prompts would lose to sampling noise *alone*.

    Cross-prompt coverage falls for two reasons and they must not be confused:
    the two prompts genuinely prefer different experts, or the top-k drawn from
    ~1000 selections over 256 experts is simply a noisy estimate of one prompt's
    own preference. This measures the second in isolation — a fresh sample of
    the same size from *this prompt's own* distribution, scored against another
    such sample. Identical true distribution by construction, so whatever
    coverage is lost here is noise.

    Cross-prompt at or above this figure means the hot set is global. Below it
    means the prompts really do route differently.
    """
    p = counts / counts.sum(axis=1, keepdims=True)
    draws = counts.sum(axis=1)

    def resample():
        return np.stack([RNG.multinomial(int(n), p[i]) for i, n in enumerate(draws)])

    return float(np.mean([coverage(topk_mask(resample(), k), resample()) for _ in range(trials)]))


def uniform_null(per_layer_draws, k, trials=200):
    """Top-k coverage a *uniform* router would show at this sample size."""
    out = []
    for _ in range(trials):
        sim = np.stack(
            [RNG.multinomial(int(n), np.full(N_EXPERT, 1 / N_EXPERT)) for n in per_layer_draws]
        )
        out.append(coverage(topk_mask(sim, k), sim))
    return float(np.mean(out))


def main():
    paths = sorted(glob.glob(os.path.join(CSV, "*.csv")))
    if not paths:
        sys.exit("no captures in " + CSV)
    names = [os.path.basename(p)[:-4] for p in paths]
    data = {n: load(p) for n, p in zip(names, paths)}

    print(f"captures: {len(names)} — {', '.join(names)}")
    print(f"layers {HASH_LAYERS}..  ({next(iter(data.values())).shape[0]} learned-gating layers)\n")

    print("=== 1. sample size, and what a uniform router would score at it ===")
    print(f"{'prompt':<10} {'sel/layer':>9} " + " ".join(f"{'top-'+str(k):>16}" for k in KS))
    print(f"{'':<10} {'':>9} " + " ".join(f"{'obs / null':>16}" for _ in KS))
    nulls = {}
    for n in names:
        c = data[n]
        draws = c.sum(axis=1)
        row = []
        for k in KS:
            obs = coverage(topk_mask(c, k), c)
            key = (int(draws.mean()), k)
            if key not in nulls:
                nulls[key] = uniform_null(draws, k)
            row.append(f"{obs*100:6.1f} / {nulls[key]*100:5.1f}")
        print(f"{n:<10} {int(draws.mean()):>9} " + " ".join(f"{r:>16}" for r in row))
    print("\n  'null' is the same number of draws from a uniform router, 200 trials.")
    print("  obs == null would mean the observed skew is entirely sample size.\n")

    print("=== 2. cross-prompt: hot set pinned from ROW, hit rate on COLUMN (top-64) ===")
    k = 64
    masks = {n: topk_mask(data[n], k) for n in names}
    header = "pin/run"
    print(f"{header:<10} " + " ".join(f"{n:>9}" for n in names))
    for a in names:
        cells = [f"{coverage(masks[a], data[b])*100:8.1f}%" for b in names]
        print(f"{a:<10} " + " ".join(f"{c:>9}" for c in cells))
    floor = k / N_EXPERT * 100
    print(f"\n  diagonal is in-sample (the optimistic v0.0.2 number).")
    print(f"  a random {k}-expert set would score {floor:.1f}%. That is the floor.\n")

    print("=== 3. same-domain vs different-domain, against the noise ceiling ===")
    print("  Read left to right: floor is a random cache, ceiling is what an")
    print("  identical router would score at this sample size. Where the")
    print("  same/diff columns sit between them is the answer to R0.")
    dom = {n: n.rsplit("_", 1)[0] for n in names}
    print(
        f"\n{'K':>4} {'floor':>8} {'diff domain':>13} {'same domain':>13}"
        f" {'noise ceiling':>15} {'in-sample':>11}"
    )
    for k in KS:
        ms = {n: topk_mask(data[n], k) for n in names}
        same, diff, self_ = [], [], []
        for a in names:
            for b in names:
                cov = coverage(ms[a], data[b])
                if a == b:
                    self_.append(cov)
                elif dom[a] == dom[b]:
                    same.append(cov)
                else:
                    diff.append(cov)
        ceil = np.mean([noise_ceiling(data[n], k) for n in names])
        print(
            f"{k:>4} {k/N_EXPERT*100:7.1f}% {np.mean(diff)*100:12.1f}%"
            f" {np.mean(same)*100:12.1f}% {ceil*100:14.1f}% {np.mean(self_)*100:10.1f}%"
        )

    # farsi_b is a *coding* question written in Persian, so topic and language
    # are crossed rather than confounded: if the hot set tracks subject matter,
    # farsi_b should resemble the English coding prompts; if it tracks the
    # script the tokenizer emits, it should resemble farsi_a instead.
    LABEL = {
        "code_a": ("en", "code"),
        "code_b": ("en", "code"),
        "prose_a": ("en", "prose"),
        "prose_b": ("en", "prose"),
        "math_a": ("en", "math"),
        "math_b": ("en", "math"),
        "farsi_a": ("fa", "prose"),
        "farsi_b": ("fa", "code"),
    }
    if all(n in LABEL for n in names):
        print("\n=== 3b. is it the topic or the language? (top-64, off-diagonal) ===")
        k = 64
        ms = {n: topk_mask(data[n], k) for n in names}
        buckets = {}
        for a in names:
            for b in names:
                if a == b:
                    continue
                la, ta = LABEL[a]
                lb, tb = LABEL[b]
                key = (
                    "same topic" if ta == tb else "diff topic",
                    "same lang" if la == lb else "diff lang",
                )
                buckets.setdefault(key, []).append(coverage(ms[a], data[b]))
        print(f"{'':<12} {'same lang':>11} {'diff lang':>11}")
        for t in ("same topic", "diff topic"):
            cells = []
            for l in ("same lang", "diff lang"):
                v = buckets.get((t, l))
                cells.append(f"{np.mean(v)*100:10.1f}%" if v else f"{'--':>11}")
            print(f"{t:<12} " + " ".join(cells))

    print("\n=== 4. leave-one-out: pin a GLOBAL hot set, hit rate on the held-out prompt ===")
    print("  This is the number a shipped cache would actually achieve.")
    print(f"{'held out':<10} " + " ".join(f"{'top-'+str(k):>9}" for k in KS))
    loo = {k: [] for k in KS}
    for b in names:
        pooled = sum(data[a] for a in names if a != b)
        cells = []
        for k in KS:
            cov = coverage(topk_mask(pooled, k), data[b])
            loo[k].append(cov)
            cells.append(f"{cov*100:8.1f}%")
        print(f"{b:<10} " + " ".join(f"{c:>9}" for c in cells))
    print(f"{'MEAN':<10} " + " ".join(f"{np.mean(loo[k])*100:8.1f}%" for k in KS))

    print("\n=== 5. what that hit rate is worth, on this laptop's measured drive ===")
    # 3.21 GiB of routed experts per token, 2.37 GiB/s measured NVMe.
    # Bytes still read = miss rate x 3.21 GiB. Disk floor only; compute is extra.
    #
    # Two columns, because they bracket the two cache designs:
    #   static  — pin a hot set chosen in advance. That is leave-one-out, and it
    #             is what "pin the global hot set" would actually deliver.
    #   in-prompt — the same prompt's own top-k. No cache reaches this (it would
    #             have to know the prompt's routing before running it), so it is
    #             an upper bound on any adaptive policy, not a target.
    gib_tok, drive = 3.21, 2.37

    def floor_tps(hit):
        miss = gib_tok * (1 - hit)
        return drive / miss if miss > 1e-9 else float("inf")

    insample = {k: float(np.mean([coverage(topk_mask(data[n], k), data[n]) for n in names])) for k in KS}
    print(
        f"{'top-K':>6} {'GiB cache':>10} {'static hit':>11} {'floor':>11}"
        f" {'in-prompt hit':>14} {'ceiling':>11}"
    )
    for k in KS:
        gib = 137.06 * k / N_EXPERT
        s, a = float(np.mean(loo[k])), insample[k]
        print(
            f"{k:>6} {gib:9.2f} {s*100:10.1f}% {floor_tps(s):8.2f} t/s"
            f" {a*100:13.1f}% {floor_tps(a):8.2f} t/s"
        )

    print("\n=== 6. what 20 tok/s would actually require ===")
    need = 1 - (drive / 20.0) / gib_tok
    print(f"  20 tok/s on a {drive} GiB/s drive needs a {need*100:.1f}% hit rate.")
    for label, series in (("static (pinned)", {k: float(np.mean(loo[k])) for k in KS}),
                          ("in-prompt ceiling", insample)):
        reach = [k for k in KS if series[k] >= need]
        if reach:
            k = min(reach)
            print(f"  {label:<20} reaches it at top-{k} = {137.06*k/N_EXPERT:.1f} GiB of cache.")
        else:
            best = max(KS)
            print(
                f"  {label:<20} does NOT reach it: top-{best} ({137.06*best/N_EXPERT:.1f} GiB)"
                f" gives {series[best]*100:.1f}%."
            )
    print("\n  Cache sizes above free RAM are not reachable on this machine; they")
    print("  describe the desktop claim, not this laptop.")


if __name__ == "__main__":
    main()
