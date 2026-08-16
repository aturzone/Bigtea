#!/usr/bin/env python3
"""Turn `assets/logo.svg` into the embedded bitmap the CLI banner prints.

Run this by hand when the logo changes; the output is committed. It exists so
the banner costs the crate **nothing**: no SVG parser, no runtime dependency,
no build script -- this workspace has zero external dependencies and the banner
was not going to be the first one.

The logo is 43 closed paths built only from `M`, `C` and `Z`, each with a solid
fill and a `translate` transform, so a full SVG implementation is not needed and
would be the wrong thing to write. What is implemented is exactly that subset:
cubic Beziers flattened to segments, filled by scanline with the nonzero winding
rule, painted in document order, supersampled and box-downsampled.

    python tools/rasterise-logo.py

Writes `crates/chaos-arch/src/logo_bitmap.rs`.
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "assets" / "logo.svg"

# Cells are half-blocks, so one cell holds two vertical pixels and the aspect
# ratio comes out square. This is the MASTER size: `banner.rs` box-filters it
# down to whatever the terminal has room for. 56 is where the eye, the hands and
# the individual rays all survive -- below about 40 the centre collapses into a
# grey blob, which was checked by rendering candidates to PNG and looking at
# them rather than by guessing.
OUT_W = int(os.environ.get("LOGO_W", 56))
OUT_H = int(os.environ.get("LOGO_H", 54))
SS = int(os.environ.get("LOGO_SS", 12))  # supersample factor per axis


def parse_paths(svg_text):
    """Every `<path>` as (list of subpaths, (r, g, b)), in document order."""
    out = []
    for m in re.finditer(r"<path\b([^>]*)/>", svg_text):
        attrs = m.group(1)
        d = re.search(r'd="([^"]*)"', attrs)
        fill = re.search(r'fill="#([0-9A-Fa-f]{6})"', attrs)
        tr = re.search(r"translate\(([-\d.]+),([-\d.]+)\)", attrs)
        if not d or not fill:
            continue
        tx, ty = (float(tr.group(1)), float(tr.group(2))) if tr else (0.0, 0.0)
        rgb = tuple(int(fill.group(1)[i : i + 2], 16) for i in (0, 2, 4))
        out.append((flatten(d.group(1), tx, ty), rgb))
    return out


def flatten(d, tx, ty):
    """`d` as a list of closed polygons in absolute device coordinates."""
    toks = re.findall(r"[MCZmcz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    polys, cur = [], []
    i, cx, cy, sx, sy = 0, 0.0, 0.0, 0.0, 0.0
    cmd = None
    while i < len(toks):
        t = toks[i]
        if t in "MCZmcz":
            cmd = t
            i += 1
            if cmd in "Zz":
                if len(cur) > 2:
                    polys.append(cur)
                cur, cx, cy = [], sx, sy
            continue
        if cmd is None:
            i += 1
            continue
        if cmd in "Mm":
            x, y = float(toks[i]), float(toks[i + 1])
            i += 2
            if cmd == "m":
                x, y = cx + x, cy + y
            if len(cur) > 2:
                polys.append(cur)
            cx, cy = x, y
            sx, sy = x, y
            cur = [(x + tx, y + ty)]
            cmd = "L" if cmd == "M" else "l"  # implicit lineto after moveto
        elif cmd in "Ll":
            x, y = float(toks[i]), float(toks[i + 1])
            i += 2
            if cmd == "l":
                x, y = cx + x, cy + y
            cx, cy = x, y
            cur.append((x + tx, y + ty))
        elif cmd in "Cc":
            vals = [float(v) for v in toks[i : i + 6]]
            i += 6
            if cmd == "c":
                vals = [
                    cx + vals[0], cy + vals[1],
                    cx + vals[2], cy + vals[3],
                    cx + vals[4], cy + vals[5],
                ]
            x1, y1, x2, y2, x, y = vals
            # 16 segments is well past the point where more changes a pixel at
            # this output size; the curves are at most a few hundred units long.
            for s in range(1, 17):
                u = s / 16.0
                v = 1.0 - u
                bx = v * v * v * cx + 3 * v * v * u * x1 + 3 * v * u * u * x2 + u * u * u * x
                by = v * v * v * cy + 3 * v * v * u * y1 + 3 * v * u * u * y2 + u * u * u * y
                cur.append((bx + tx, by + ty))
            cx, cy = x, y
        else:
            i += 1
    if len(cur) > 2:
        polys.append(cur)
    return polys


def rasterise(paths, w, h, sw, sh, scale_x, scale_y):
    """Painter's algorithm at supersample resolution. Returns a row-major grid."""
    grid = [[(255, 255, 255)] * sw for _ in range(sh)]
    for polys, rgb in paths:
        # Bucket the edges by the scanline range they cross so each row only
        # looks at edges that can possibly cross it. Without this the script
        # takes minutes instead of seconds.
        edges = []
        for poly in polys:
            n = len(poly)
            for k in range(n):
                x0, y0 = poly[k]
                x1, y1 = poly[(k + 1) % n]
                if y0 == y1:
                    continue
                edges.append((x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y))
        if not edges:
            continue
        buckets = [[] for _ in range(sh)]
        for e in edges:
            lo = max(0, int(min(e[1], e[3])))
            hi = min(sh - 1, int(max(e[1], e[3])) + 1)
            for row in range(lo, hi + 1):
                buckets[row].append(e)
        for row in range(sh):
            if not buckets[row]:
                continue
            yc = row + 0.5
            xs = []
            for x0, y0, x1, y1 in buckets[row]:
                if (y0 <= yc < y1) or (y1 <= yc < y0):
                    t = (yc - y0) / (y1 - y0)
                    xs.append((x0 + t * (x1 - x0), 1 if y1 > y0 else -1))
            if not xs:
                continue
            xs.sort()
            wind = 0
            line = grid[row]
            for k in range(len(xs) - 1):
                wind += xs[k][1]
                if wind == 0:
                    continue
                a = max(0, int(xs[k][0] + 0.5))
                b = min(sw, int(xs[k + 1][0] + 0.5))
                for px in range(a, b):
                    line[px] = rgb
    # Box-downsample to the output size.
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            r = g = b = n = 0
            for sy in range(y * SS, (y + 1) * SS):
                line = grid[sy]
                for sx in range(x * SS, (x + 1) * SS):
                    pr, pg, pb = line[sx]
                    r += pr
                    g += pg
                    b += pb
                    n += 1
            row.append((r // n, g // n, b // n))
        out.append(row)
    return out


def main():
    text = SVG.read_text(encoding="utf-8")
    vw = float(re.search(r'width="([\d.]+)"', text).group(1))
    vh = float(re.search(r'height="([\d.]+)"', text).group(1))
    paths = parse_paths(text)
    sw, sh = OUT_W * SS, OUT_H * SS
    grid = rasterise(paths, OUT_W, OUT_H, sw, sh, sw / vw, sh / vh)

    # The logo is black on white, so one byte of luminance per pixel is lossless
    # here and a third of the size of RGB.
    gray = [
        [(px[0] * 299 + px[1] * 587 + px[2] * 114) // 1000 for px in row]
        for row in grid
    ]

    # An eyeball check on the way past, so a broken render is visible in the
    # terminal that ran the script rather than in the shipped binary.
    ramp = " .:-=+*#%@"
    for row in gray:
        sys.stdout.write("".join(ramp[min(9, (255 - v) * 10 // 256)] for v in row) + "\n")

    body = "\n".join(
        "    " + ", ".join(f"0x{v:02x}" for v in row) + ","
        for row in gray
    )
    out = ROOT / "crates" / "chaos-arch" / "src" / "logo_bitmap.rs"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "// GENERATED by tools/rasterise-logo.py from assets/logo.svg -- do not edit.\n"
        "//\n"
        "// One byte of luminance per pixel, row-major, white background. The logo\n"
        "// is black on white, so luminance is lossless here and a third the size\n"
        "// of RGB. Printed with Unicode half-blocks, two pixels to a cell.\n"
        f"pub const LOGO_W: usize = {OUT_W};\n"
        f"pub const LOGO_H: usize = {OUT_H};\n"
        "// `rustfmt::skip` because otherwise regenerating this file fails CI:\n"
        "// rustfmt reflows the array to its own width and the next run of the\n"
        "// script puts it straight back, so the two disagree forever.\n"
        "#[rustfmt::skip]\n"
        f"pub const LOGO: [u8; {OUT_W * OUT_H}] = [\n{body}\n];\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out} ({OUT_W}x{OUT_H}, {OUT_W * OUT_H} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
