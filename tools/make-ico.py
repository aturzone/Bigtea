"""Build `assets/chaos.ico` from `assets/logo.svg`.

Windows shows a blank default icon for an executable that carries none, which
is what "empty shapes" looked like on the setup, the app, the taskbar and the
Start Menu entry.

**Every size is rasterised from the vector at its own resolution.** Windows does
not downsample well: the shell asks for 16 px for a title bar and 256 px for the
Alt-Tab view, and a single 256 squeezed to 16 turns this logo -- which is mostly
fine radiating lines -- into grey mush. Nine sizes, nine renders.

Separate from `rasterise-logo.py` rather than bolted onto it: that script owns
the terminal bitmap and the README image, and its `main()` is already doing two
jobs. This imports its geometry rather than copying it, so there is still one
definition of how the logo is drawn.

No dependencies, like everything else here -- `struct` and `zlib` are standard
library, and the ICO container is a header, a directory and the PNGs.

    python tools/make-ico.py
"""

import io
import pathlib
import runpy
import struct
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# What Windows actually asks for. 256 is the shell's large-icon view; 20 and 40
# are the 125% and 250% scalings of 16 and 32, which high-DPI machines request
# and which look soft if they have to be interpolated.
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def load_rasteriser():
    """Pull the geometry out of `rasterise-logo.py` without running its `main`.

    The filename has a hyphen, so it cannot be imported as a module; `runpy`
    with `run_name` set to something other than `"__main__"` executes the
    definitions and leaves the `if __name__ == "__main__"` block alone.
    """
    ns = runpy.run_path(str(ROOT / "tools" / "rasterise-logo.py"), run_name="chaos_logo")
    missing = [n for n in ("parse_paths", "rasterise", "ink_box", "SVG") if n not in ns]
    if missing:
        sys.exit(f"rasterise-logo.py no longer exports {missing}; make-ico.py needs it")
    return ns


def png_bytes(grid):
    """A truecolour PNG in memory. Same encoder shape as the README image."""
    h = len(grid)
    w = len(grid[0])
    raw = b"".join(bytes([0]) + bytes(c for px in row for c in px) for row in grid)

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    out = io.BytesIO()
    out.write(bytes([137, 80, 78, 71, 13, 10, 26, 10]))
    out.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
    out.write(chunk(b"IDAT", zlib.compress(raw, 9)))
    out.write(chunk(b"IEND", b""))
    return out.getvalue()


def build(path, render):
    """Write the .ico. `render(n)` returns an n-by-n grid of (r, g, b)."""
    images = [(n, png_bytes(render(n))) for n in SIZES]

    out = io.BytesIO()
    # ICONDIR: reserved, type 1 = icon, count.
    out.write(struct.pack("<HHH", 0, 1, len(images)))
    offset = 6 + 16 * len(images)
    for n, data in images:
        # **256 is written as 0.** The width and height fields are single bytes,
        # so the format spells 256 as zero; writing 255 instead produces a file
        # Explorer accepts and then never shows at large sizes.
        b = 0 if n >= 256 else n
        out.write(struct.pack("<BBBBHHII", b, b, 0, 0, 1, 32, len(data), offset))
        offset += len(data)
    for _, data in images:
        out.write(data)
    path.write_bytes(out.getvalue())
    return len(images), offset


def main():
    ns = load_rasteriser()
    paths = ns["parse_paths"](ns["SVG"].read_text(encoding="utf-8"))
    bx, by, side = ns["ink_box"](paths)
    rasterise = ns["rasterise"]

    def render(px):
        # Supersample 3x and box down, as the README image does -- an icon is
        # small enough that aliasing on these thin rays is the whole difference
        # between a mark and a smudge.
        return rasterise(
            paths, px, px, px * 3, px * 3, px * 3 / side, px * 3 / side, ss=3, origin=(bx, by)
        )

    out = ROOT / "assets" / "chaos.ico"
    n, size = build(out, render)
    print(f"wrote {out} ({n} sizes {SIZES[0]}-{SIZES[-1]}, {size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
