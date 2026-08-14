"""Compare Bigtea's Jinja rendering against `llama.cpp --jinja`, per container.

The family matcher is verified against llama.cpp's *hardcoded* renderer. This
compares the other pair: our template evaluation against llama.cpp's, which is
the only thing that can settle a disagreement between the two paths.

It already has: on Llama-3.2 both hardcoded renderers drop the "Cutting
Knowledge Date" preamble that the model's own template emits, and only running
`--jinja` showed it.

    python scripts/jinja-vs-llamacpp.py [model.gguf ...]

With no arguments it sweeps every container under C:/Projects/models.
"""

import re
import subprocess
import sys
from pathlib import Path

LLAMA = Path("C:/Projects/llamacpp-unsloth/build/bin/llama-completion.exe")
MODELS = Path("C:/Projects/models")
SYSTEM, USER = "SYS", "HI"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PREFIX = re.compile(r"^\d+\.\d+\.\d+\.\d+ I ", re.M)
TOKEN = re.compile(r"^\s*\d+ -> '(.*)'\s*$", re.S)


def rendered(model: Path, jinja: bool) -> str | None:
    """The exact prompt llama.cpp builds, reconstructed from --verbose-prompt.

    Token by token rather than from a log line, because no log line prints the
    formatted prompt -- and the tokens are what the model actually sees, which
    is the thing worth comparing.
    """
    out = subprocess.run(
        [str(LLAMA), "-m", str(model), "--jinja" if jinja else "--no-jinja",
         "-sys", SYSTEM, "-p", USER, "-n", "1", "--temp", "0", "-st",
         "--verbose-prompt"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    text = ANSI.sub("", (out.stdout or "") + (out.stderr or ""))
    pieces = [m.group(1) for c in PREFIX.split(text) if (m := TOKEN.match(c))]
    return "".join(pieces) if pieces else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    args = sys.argv[1:]
    models = [Path(a) for a in args] if args else sorted(
        p for d in MODELS.iterdir() if d.is_dir() for p in d.glob("*.gguf")
    )
    # One shard is enough: the template is metadata, identical across shards,
    # and loading a 144 GB model five times to read the same string is a waste.
    seen: set[str] = set()
    worst = 0
    for m in models:
        stem = re.sub(r"-\d{5}-of-\d{5}", "", m.stem)
        if stem in seen:
            continue
        seen.add(stem)
        hard, soft = rendered(m, False), rendered(m, True)
        if hard is None or soft is None:
            print(f"skip  {m.name}: llama.cpp declined to load it")
            continue
        verdict = "same" if hard == soft else "DIFFER"
        if hard != soft:
            worst = 1
        print(f"{verdict:6} {m.name}")
        if hard != soft:
            print(f"  --no-jinja: {hard!r}")
            print(f"  --jinja   : {soft!r}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
