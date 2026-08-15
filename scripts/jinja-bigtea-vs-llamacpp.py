"""Compare **Bigtea's** chat framing against llama.cpp's, both paths, per container.

`jinja-vs-llamacpp.py` compares llama.cpp's hardcoded renderer against
llama.cpp's own Jinja — it never runs Bigtea, despite what its docstring said.
That measurement is worth having (the reference disagrees with itself on 5 of 18
containers) but it does not check our engine at all, and it was being cited as
though it did.

This runs the four-way that actually settles it:

    bigtea --jinja   vs  llama.cpp --jinja      does OUR Jinja match THEIRS?
    bigtea           vs  llama.cpp --no-jinja   does our family matcher match?

**Token IDs, not rendered text.** The tokens are what the model sees, and two
renderings that differ only in a trailing newline the tokenizer drops are not a
difference worth failing on — while two that tokenize apart are, however similar
they look. It also avoids reconstructing text from a log, which is what the older
script does.

    python scripts/jinja-bigtea-vs-llamacpp.py [model.gguf ...]

With no arguments it sweeps every container under C:/Projects/models.
"""

import re
import subprocess
import sys
from pathlib import Path

LLAMA = Path("C:/Projects/llamacpp-unsloth/build/bin/llama-completion.exe")
BIGTEA = Path("./target/release/bigtea-run.exe")
MODELS = Path("C:/Projects/models")
SYSTEM, USER = "SYS", "HI"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# llama.cpp: `0.00.477.182 I    450 -> ' The'`. Anchored on the arrow rather
# than on the quotes, because token texts contain both quotes and apostrophes.
LLAMA_ID = re.compile(r"^[\d.]+ I\s+(\d+) -> ", re.M)
# bigtea: `prompt     6 tokens: [1, 450, 7483, 310, 3444, 338]`
BIGTEA_IDS = re.compile(r"^prompt\s+\d+ tokens: \[([\d, ]*)\]", re.M)


def run(cmd: list[str]) -> str:
    out = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return ANSI.sub("", (out.stdout or "") + (out.stderr or ""))


def llama_tokens(model: Path, jinja: bool) -> list[int] | None:
    text = run([
        str(LLAMA), "-m", str(model), "--jinja" if jinja else "--no-jinja",
        "-sys", SYSTEM, "-p", USER, "-n", "1", "--temp", "0", "-st",
        "--verbose-prompt",
    ])
    ids = [int(m) for m in LLAMA_ID.findall(text)]
    return ids or None


def bigtea_tokens(model: Path, jinja: bool) -> list[int] | None:
    cmd = [
        str(BIGTEA), "-m", str(model), "-sys", SYSTEM, "-p", USER,
        "-n", "1", "--temp", "0", "--force", "-cnv", "--verbose-prompt",
    ]
    if jinja:
        cmd.append("--jinja")
    m = BIGTEA_IDS.search(run(cmd))
    if not m or not m.group(1).strip():
        return None
    return [int(x) for x in m.group(1).split(",")]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    args = sys.argv[1:]
    models = [Path(a) for a in args] if args else sorted(
        p for d in MODELS.iterdir() if d.is_dir() for p in d.glob("*.gguf")
    )
    # **Models with NO chat template are not comparable through this script and
    # are skipped rather than counted as differences.**
    #
    # `llama-completion -sys X -p Y` on such a container does RAW COMPLETION --
    # it emits the user text and nothing else, because there is no template to
    # apply. Bigtea's `-cnv` applies its neutral framing. Reporting that as a
    # disagreement compares our chat path against the reference's completion
    # path, and it counted three models as broken that are not.
    #
    # llama.cpp's actual fallback for a missing template is ChatML
    # (`common/chat.cpp`: `template_default` "always set (defaults to chatml)"),
    # but it lives on the conversation path, which this script does not drive.
    # Comparing against it needs `llama-server` or an interactive session.
    no_template = {"OLMo-1B", "starcoder2-3b", "all-MiniLM-L6-v2"}

    seen: set[str] = set()
    worst = 0
    agree = differ = skipped = 0
    for m in models:
        if any(m.stem.startswith(t) for t in no_template):
            print(f"skip   {m.name}: no chat template -- see the note in this script")
            skipped += 1
            continue
        # One shard is enough: the template is metadata and identical across
        # shards, and loading a 144 GB model five times to read one string is a
        # waste.
        stem = re.sub(r"-\d{5}-of-\d{5}", "", m.stem)
        if stem in seen:
            continue
        seen.add(stem)

        rows = []
        for label, jinja in (("jinja  ", True), ("family ", False)):
            ours, theirs = bigtea_tokens(m, jinja), llama_tokens(m, jinja)
            if ours is None or theirs is None:
                rows.append((label, None, ours, theirs))
            else:
                rows.append((label, ours == theirs, ours, theirs))

        if all(r[1] is None for r in rows):
            print(f"skip   {m.name}: one of the engines declined to load it")
            skipped += 1
            continue

        bad = [r for r in rows if r[1] is False]
        if not bad:
            print(f"same   {m.name}")
            agree += 1
            continue

        differ += 1
        worst = 1
        print(f"DIFFER {m.name}")
        for label, ok, ours, theirs in rows:
            if ok is False:
                print(f"  {label} bigtea   : {ours}")
                print(f"  {label} llama.cpp: {theirs}")
            elif ok is None:
                print(f"  {label} (one engine declined)")

    print(f"\n{agree} agree, {differ} differ, {skipped} skipped")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
