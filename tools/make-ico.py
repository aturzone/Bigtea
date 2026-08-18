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
    """An **RGBA** PNG in memory -- colour type 6, not 2.

    The alpha channel is the whole point of the rounded corner. A truecolour
    PNG has nowhere to say "not the icon", so the corners have to be filled with
    something, and whatever is chosen shows as a square: a white fill made the
    icon a white tile and the rounding invisible. Windows reads the alpha in a
    PNG-compressed ICO entry directly.
    """
    h = len(grid)
    w = len(grid[0])
    raw = b"".join(bytes([0]) + bytes(c for px in row for c in px) for row in grid)

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    out = io.BytesIO()
    out.write(bytes([137, 80, 78, 71, 13, 10, 26, 10]))
    out.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)))
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


# The tile the mark sits on.
#
# Hermes' own `BrandMark` is "the mark on a tile, softly rounded, identical in
# light and dark", and Atur asked for a curved square -- a bare mark on
# transparency reads as a smudge on a light taskbar and on a dark one.
#
# **The blue is the ground, not the mark.** That was Atur's correction: `#0000F2`
# is what Hermes puts *behind* a wordmark, so here it is the tile and the mark is
# knocked out of it in white. The art in the SVG is black on white, so it is
# inverted on the way in.
TILE = (0x00, 0x00, 0xF2)
MARK = (255, 255, 255)
INK_INSET = 0.16  # the mark occupies the middle 68% of the tile
RADIUS_FRAC = 0.22  # corner radius as a fraction of the side; iOS-ish, not a circle


def rounded_mask(px, ss=4):
    """Coverage of a rounded square, 0..1 per pixel, supersampled.

    Antialiased by counting subsamples: a hard-edged corner at 16 px is a
    staircase, and the icon is seen at 16 px more often than at any other size.
    """
    r = RADIUS_FRAC * px
    mask = []
    for y in range(px):
        row = []
        for x in range(px):
            hits = 0
            for sy in range(ss):
                for sx in range(ss):
                    fx = x + (sx + 0.5) / ss
                    fy = y + (sy + 0.5) / ss
                    # Distance into the nearest corner's circle, or inside the
                    # straight edges.
                    cx = min(max(fx, r), px - r)
                    cy = min(max(fy, r), px - r)
                    if (fx - cx) ** 2 + (fy - cy) ** 2 <= r * r:
                        hits += 1
            row.append(hits / (ss * ss))
        mask.append(row)
    return mask


def main():
    ns = load_rasteriser()
    paths = ns["parse_paths"](ns["SVG"].read_text(encoding="utf-8"))
    bx, by, side = ns["ink_box"](paths)
    rasterise = ns["rasterise"]

    def render(px):
        # The mark, rendered smaller than the tile so it has margin, then
        # composited onto a rounded white square.
        inner = max(1, int(round(px * (1 - 2 * INK_INSET))))
        art = rasterise(
            paths, inner, inner, inner * 3, inner * 3,
            inner * 3 / side, inner * 3 / side, ss=3, origin=(bx, by),
        )
        mask = rounded_mask(px)
        off = (px - inner) // 2
        grid = []
        for y in range(px):
            row = []
            for x in range(px):
                iy, ix = y - off, x - off
                if 0 <= iy < inner and 0 <= ix < inner:
                    r, g, b = art[iy][ix]
                    # Ink coverage: the source is black on white, and the mark
                    # is drawn in white on the tile, so it inverts here.
                    ink = 1.0 - (r * 299 + g * 587 + b * 114) / 1000.0 / 255.0
                else:
                    ink = 0.0
                colour = tuple(
                    int(TILE[c] + (MARK[c] - TILE[c]) * ink) for c in range(3)
                )
                # Alpha is the rounded square: outside it the icon is simply not
                # there, which is what makes the corner a curve rather than a
                # differently-coloured square.
                row.append(colour + (int(round(mask[y][x] * 255)),))
            grid.append(row)
        return grid

    out = ROOT / "assets" / "chaos.ico"
    n, size = build(out, render)
    print(f"wrote {out} ({n} sizes {SIZES[0]}-{SIZES[-1]}, {size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
