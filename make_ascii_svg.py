#!/usr/bin/env python3
"""
make_ascii_svg.py -- turn an image into a self-typing, monochrome ASCII portrait
SVG that animates inside a GitHub profile README.

No JavaScript, no third-party stats service, no token: the animation is pure
SMIL inside the SVG, which GitHub renders inside <img>.

Adapted from the classic technique, swapping the heavy photo-oriented deps for
numpy/PIL-only equivalents:
  * rembg  -> colour-distance thresholding against the detected background colour
  * opencv -> PIL median filter + a small numpy CLAHE implementation

Usage:
    python make_ascii_svg.py photo.jpg ascii.svg

    # luminous subjects (glowing objects, backlit scenes, this black hole):
    python make_ascii_svg.py image.png ascii.svg --gamma 0.5
"""

import argparse

import numpy as np
from PIL import Image, ImageFilter

# ---- Tunables ----------------------------------------------------------------
RAMP = " .`:-=+*cs#%@"   # bright/sparse -> dark/dense; leading space clears bg
COLS = 130               # characters per row
CLAHE_CLIP = 2.2         # per-tile contrast; higher amplifies skin/texture noise
GAMMA = 1.35             # >1: only real shadow draws (portraits). Luminous
                        # subjects (glow, backlight, this black hole) want <1
                        # so the glow renders as characters, e.g. --gamma 0.8
CROP_BOTTOM = 0.0        # trim this fraction off the bottom
BG_COLOR = None          # auto-detected from the image border when None
BG_TOL = 40              # max RGB distance from BG_COLOR to be treated as bg
BG_DILATE = 2            # grow the bg mask by N px (absorbs vignette edge bands
                        # and stray specks that sit just outside the tolerance).
                        # Note: this also erodes thin subject features by N px.
MEDIAN = 3               # median filter kernel size (kills 1px specks/stars)
FG_LIGHT = "#6e7681"     # grey, readable on GitHub light mode
FG_DARK = "#c9d1d9"      # light grey on GitHub dark mode
CHAR_W = 7.74
FONT_SIZE = 12.9
LINE_H = 15
ROW_DELAY = 0.09         # seconds between rows
# ------------------------------------------------------------------------------


def detect_bg(rgb):
    """Estimate the background colour as the dominant quantized colour.

    Works when the background is a single, fairly uniform colour (pixel art,
    product shots on a plain backdrop). For photos with busy backgrounds use
    rembg instead, as in the original guide.
    """
    q = (rgb.astype(np.int16) // 16 * 16)
    vals, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    best = vals[np.argmax(counts)]
    mask = (q.reshape(-1, 3) == best).all(axis=1)
    return rgb.reshape(-1, 3)[mask].mean(axis=0)


def clahe(gray, clip_limit=CLAHE_CLIP, grid=(8, 8)):
    """Contrast-limited adaptive histogram equalisation (pure numpy).

    Takes 0-255 uint8 grayscale, returns float32 in the same range. Each tile
    gets its own clipped cumulative histogram; neighbouring tiles are blended
    bilinearly so there are no visible tile seams.
    """
    h, w = gray.shape
    gh, gw = grid
    tile_h, tile_w = max(1, h // gh), max(1, w // gw)
    gh, gw = h // tile_h, w // tile_w          # fit an exact grid
    # pad with edge rows/cols so every pixel is equalized, then crop back
    pad = (gh * tile_h - h, gw * tile_w - w)
    work = gray
    if pad[0] or pad[1]:
        work = np.pad(gray, ((0, pad[0]), (0, pad[1])), mode="edge")
    g = work.astype(np.float32)

    tiles = np.zeros((gh, gw, 256), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            tile = g[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
            hist, _ = np.histogram(tile.ravel(), bins=256, range=(0, 256))
            clip = max(1, int(clip_limit * (tile_h * tile_w) / 256))
            excess = float(np.clip(hist - clip, 0, None).sum())
            hist = np.minimum(hist, clip) + excess / 256
            tiles[i, j] = np.cumsum(hist) / hist.sum() * 255.0

    ii, jj = np.indices((g.shape[0], g.shape[1]))
    ti = np.minimum(ii // tile_h, gh - 1)
    tj = np.minimum(jj // tile_w, gw - 1)
    ti1, tj1 = np.minimum(ti + 1, gh - 1), np.minimum(tj + 1, gw - 1)
    fi = np.clip((ii % tile_h) / tile_h, 0.0, 1.0)
    fj = np.clip((jj % tile_w) / tile_w, 0.0, 1.0)
    v = g[ii, jj].astype(np.int16)

    top = tiles[ti, tj, v] * (1 - fi) + tiles[ti1, tj, v] * fi
    bot = tiles[ti, tj1, v] * (1 - fi) + tiles[ti1, tj1, v] * fi
    out = top * (1 - fj) + bot * fj

    res = work.astype(np.float32).copy()
    res[: out.shape[0], : out.shape[1]] = out
    return res[:h, :w]


def _dilate(mask, iters=BG_DILATE):
    """Binary dilation (8-neighbourhood) in pure numpy."""
    m = mask
    for _ in range(iters):
        p = np.pad(m, 1, mode="edge")
        m = (m
             | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
             | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:])
    return m


def prep(path, tol=BG_TOL, clip=CLAHE_CLIP, dilate=BG_DILATE):
    """Cut the background, smooth, boost local contrast. Returns (gray, bg, bg_frac)."""
    img = Image.open(path).convert("RGB")
    if MEDIAN > 1:
        img = img.filter(ImageFilter.MedianFilter(MEDIAN))
    rgb = np.asarray(img).astype(np.float32)

    bg = BG_COLOR if BG_COLOR is not None else detect_bg(rgb)
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    bg_mask = _dilate(dist < tol, dilate or 0)

    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gray = np.clip(lum, 0, 255).astype(np.uint8).astype(np.float32)
    gray[bg_mask] = 255.0                    # bg -> blank end of the ramp
    gray = clahe(np.clip(gray, 0, 255).astype(np.uint8), clip)
    gray[bg_mask] = 255.0                    # CLAHE never moves 255; belt and braces
    return gray, bg, bg_mask.mean()


def to_lines(gray, cols=COLS, gamma=GAMMA, crop=CROP_BOTTOM):
    h, w = gray.shape
    if crop:
        gray = gray[: int(h * (1 - crop)), :]
        h, w = gray.shape
    rows = max(1, int(cols * (h / w) * 0.48))
    img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).resize(
        (cols, rows), Image.LANCZOS
    )
    px = list(img.getdata())
    n = len(RAMP)
    out = []
    for r in range(rows):
        line = "".join(
            RAMP[min(n - 1, int((1 - px[r * cols + c] / 255.0) ** gamma * n))]
            for c in range(cols)
        )
        out.append(line.rstrip())
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def build_svg(lines, out_path, cols=COLS):
    pad = 14
    width = int(cols * CHAR_W + pad * 2)
    height = len(lines) * LINE_H + pad * 2
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">',
        f'<style>.a{{fill:{FG_LIGHT}}}'
        f'@media(prefers-color-scheme:dark){{.a{{fill:{FG_DARK}}}}}</style>',
    ]
    for i, line in enumerate(lines):
        y = pad + i * LINE_H
        begin = f"{i * ROW_DELAY:.2f}s"
        end = f"{(i + 1) * ROW_DELAY:.2f}s"
        w = max(len(line), 1) * CHAR_W
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        p.append(
            f'<clipPath id="c{i}"><rect x="{pad}" y="{y}" height="{LINE_H}" width="0">'
            f'<animate attributeName="width" from="0" to="{w:.1f}" '
            f'begin="{begin}" dur="{ROW_DELAY}s" fill="freeze"/></rect></clipPath>'
        )
        p.append(
            f'<g clip-path="url(#c{i})"><text xml:space="preserve" x="{pad}" '
            f'y="{y + 11.2:.1f}" class="a" font-size="{FONT_SIZE}">{safe}</text></g>'
        )
        p.append(
            f'<rect y="{y + 1}" width="6" height="12" class="a" opacity="0">'
            f'<animate attributeName="x" from="{pad}" to="{pad + w:.1f}" '
            f'begin="{begin}" dur="{ROW_DELAY}s" fill="freeze"/>'
            f'<set attributeName="opacity" to="0.8" begin="{begin}"/>'
            f'<set attributeName="opacity" to="0" begin="{end}"/></rect>'
        )
    p.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(p))
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Make a self-typing ASCII portrait SVG.")
    ap.add_argument("src", help="input image")
    ap.add_argument("dst", nargs="?", default="ascii.svg", help="output SVG (default: ascii.svg)")
    ap.add_argument("--gamma", type=float, default=GAMMA, help="char ramp gamma (default %(default)s)")
    ap.add_argument("--cols", type=int, default=COLS, help="characters per row (default %(default)s)")
    ap.add_argument("--clip", type=float, default=CLAHE_CLIP, help="CLAHE clip limit (default %(default)s)")
    ap.add_argument("--tol", type=float, default=BG_TOL, help="bg colour distance tolerance (default %(default)s)")
    ap.add_argument("--dilate", type=int, default=BG_DILATE, help="bg mask dilation in px (default %(default)s)")
    ap.add_argument("--crop-bottom", type=float, default=CROP_BOTTOM, help="fraction trimmed off the bottom")
    args = ap.parse_args()
    args.dilate = max(0, args.dilate)

    gray, bg, bg_frac = prep(args.src, tol=args.tol, clip=args.clip, dilate=args.dilate)
    lines = to_lines(gray, cols=args.cols, gamma=args.gamma, crop=args.crop_bottom)
    print("\n".join(lines))
    build_svg(lines, args.dst, cols=args.cols)
    print(
        f"\nbg colour: {tuple(int(c) for c in bg)}  "
        f"background: {bg_frac * 100:.1f}%  rows: {len(lines)}"
    )
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
