"""Tokenize the same awkward strings in both engines, every container, ID for ID.

This exists because of Falcon3. It stores a **raw newline** at id 12, where a
byte-level BPE vocabulary stores `Ċ`; we encoded first, looked up `Ċ`, found
nothing, and dropped the token silently:

    a\\nb    ours [11, 2088, 2089]    llama.cpp [11, 2088, 12, 2089]

**It passed 8/8 parity.** None of `parity-check.sh`'s eight prompts contains a
newline. The bug reached every chat template on the model — all of which do —
and was invisible to the check that was supposed to cover exactly this class.

So the generalisation: a parity harness scored on *natural sentences* tests the
tokenizer on the easiest input it will ever see. The interesting failures live in
whitespace, control characters, combining marks, and text that ends mid-codepoint
in a naive splitter. Those are cheap to enumerate and were not being enumerated.

    python scripts/tokenizer-parity.py [model.gguf ...]

Compares token IDs, not text. Exits non-zero if any case differs.
"""

import re
import subprocess
import sys
from pathlib import Path

LLAMA = Path("C:/Projects/llamacpp-unsloth/build/bin/llama-completion.exe")
BIGTEA = Path("./target/release/bigtea-run.exe")
MODELS = Path("C:/Projects/models")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LLAMA_ID = re.compile(r"^[\d.]+ I\s+(\d+) -> ", re.M)
BIGTEA_IDS = re.compile(r"^prompt\s+\d+ tokens: \[([\d, ]*)\]", re.M)

# Each name says what would break if the case failed, so a red line explains
# itself without anyone re-deriving why the string was chosen.
CASES = [
    ("plain", "The capital of France is"),
    ("newline", "a\nb"),
    ("blank line", "a\n\nb"),
    ("tab", "a\tb"),
    ("crlf", "a\r\nb"),
    ("leading space", " hello"),
    ("double space", "a  b"),
    ("trailing space", "hello "),
    ("indented code", "def f():\n    return 1\n"),
    ("cjk", "\u4f60\u597d\u4e16\u754c"),
    ("emoji", "hi \U0001f600 there"),
    ("combining mark", "cafe\u0301"),
    ("accented", "na\u00efve r\u00e9sum\u00e9"),
    ("digits", "1234567890"),
    ("punctuation run", "!!!???...---"),
    ("mixed script", "hello \u4e16\u754c 123"),
    ("json", '{"a": [1, 2], "b": null}'),
]


def run(cmd: list[str]) -> str:
    out = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600,
    )
    return ANSI.sub("", (out.stdout or "") + (out.stderr or ""))


def llama_ids(model: Path, text: str) -> list[int] | None:
    out = run([
        str(LLAMA), "-m", str(model), "-p", text, "-n", "1", "--temp", "0",
        "--no-warmup", "-no-cnv", "--verbose-prompt",
    ])
    ids = [int(m) for m in LLAMA_ID.findall(out)]
    return ids or None


def bigtea_ids(model: Path, text: str) -> list[int] | None:
    out = run([
        str(BIGTEA), "-m", str(model), "-p", text, "-n", "1", "--temp", "0",
        "--force", "--verbose-prompt",
    ])
    m = BIGTEA_IDS.search(out)
    if not m:
        return None
    body = m.group(1).strip()
    return [int(x) for x in body.split(",")] if body else []


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    args = sys.argv[1:]
    models = [Path(a) for a in args] if args else sorted(
        p for d in MODELS.iterdir() if d.is_dir() for p in d.glob("*.gguf")
    )

    seen: set[str] = set()
    worst = 0
    for m in models:
        # One shard is enough: the vocabulary is metadata and identical across
        # them, and loading a 144 GB model five times to read it is a waste.
        stem = re.sub(r"-\d{5}-of-\d{5}", "", m.stem)
        if stem in seen:
            continue
        seen.add(stem)

        bad = []
        checked = 0
        for name, text in CASES:
            ours, theirs = bigtea_ids(m, text), llama_ids(m, text)
            if ours is None or theirs is None:
                continue
            checked += 1
            if ours != theirs:
                bad.append((name, ours, theirs))

        if checked == 0:
            print(f"skip   {m.name}: one of the engines declined to load it")
            continue
        if not bad:
            print(f"ok     {m.name}  ({checked} cases)")
            continue

        worst = 1
        print(f"DIFFER {m.name}  ({len(bad)} of {checked})")
        for name, ours, theirs in bad:
            print(f"  {name}")
            print(f"    bigtea   : {ours}")
            print(f"    llama.cpp: {theirs}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
