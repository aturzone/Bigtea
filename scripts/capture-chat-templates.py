"""Capture llama.cpp's own rendering of every chat template it knows.

The output is a fixture the Rust tests compare against, so "Chaos renders the
`gpt-oss` template correctly" means "byte-identical to what llama.cpp produced,
on this command line, on this day" rather than "it looked right to me".

Why reconstructed from `--verbose-prompt` rather than read out of
`llama-chat.cpp`: the renderer and the tokenizer disagree in ways that only
show up after tokenization (a template that emits a space before a special
token, say), and the prompt the model actually sees is the thing worth
matching. `--verbose-prompt` prints exactly that, token by token.

    python scripts/capture-chat-templates.py > crates/chaos-tokenizer/tests/chat-templates.txt

Newline tokens print as a quote, a real newline, and a quote on the next line,
so the parse cannot be line-based -- it splits on the log's timestamp prefix
instead and takes everything between the first `'` and the last.
"""

import re
import subprocess
import sys

LLAMA = "C:/Projects/llamacpp-unsloth/build/bin/llama-completion.exe"
MODEL = "C:/Projects/models/qwen2/Qwen2-0.5B-Instruct-Q4_K_M.gguf"
CHAT_CPP = "C:/Projects/llamacpp-unsloth/src/llama-chat.cpp"

# Short and distinctive, so a renderer that drops or duplicates a turn is
# obvious in the diff rather than hidden in prose.
SYSTEM = "SYS"
USER = "HI"

PREFIX = re.compile(r"^\d+\.\d+\.\d+\.\d+ I ", re.M)
TOKEN = re.compile(r"^\s*\d+ -> '(.*)'\s*$", re.S)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def names():
    """Every template name llama.cpp accepts, from its own table."""
    src = open(CHAT_CPP, encoding="utf-8").read()
    return sorted(set(re.findall(r'^\s*\{ "([^"]+)"', src, re.M)))


def render(name):
    """The exact prompt llama.cpp builds for `name`, or None if it declined."""
    out = subprocess.run(
        [LLAMA, "-m", MODEL, "--chat-template", name, "-sys", SYSTEM,
         "-p", USER, "-n", "1", "--temp", "0", "-st", "--verbose-prompt"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    text = ANSI.sub("", (out.stdout or "") + (out.stderr or ""))
    # Everything after the token dump header; chunks are separated by the log
    # prefix, and a chunk holding a token looks like `<id> -> '<text>'`.
    pieces = []
    for chunk in PREFIX.split(text):
        m = TOKEN.match(chunk)
        if m:
            pieces.append(m.group(1))
    return "".join(pieces) if pieces else None


def main():
    # DeepSeek-3 uses full-width bars (U+FF5C) and MiniCPM Chinese role tags;
    # Windows' default cp1252 stdout dies on both, mid-capture, after the file
    # already looks half-written.
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print("# llama.cpp chat-template renderings, captured by")
    print("# scripts/capture-chat-templates.py. Do not edit by hand.")
    print(f"# model={MODEL.rsplit('/', 1)[-1]} system={SYSTEM!r} user={USER!r}")
    print("# add_generation_prompt is on (llama.cpp's -st conversation default).")
    print("# One record per template: NAME, then the rendering with newlines")
    print("# escaped as \\n so a record is exactly one line.")
    ok = 0
    for name in names():
        r = render(name)
        if r is None:
            print(f"!{name}\tDECLINED", flush=True)
            continue
        ok += 1
        # Escape the rendering first, then join -- escaping the joined line
        # would eat the separator.
        body = r.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
        print(f"{name}\t{body}", flush=True)
    print(f"# {ok} templates captured", file=sys.stderr)


if __name__ == "__main__":
    main()
