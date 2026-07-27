#!/usr/bin/env python3
"""
make_collage.py - tile all 332 labeled panels from test_caries_full/testset/
into a single grid image.

Panels have variable heights (image + findings text). We normalize to a fixed
tile size by resizing longest side to TILE, then centering on a TILE x TILE
canvas so nothing is stretched.

    python make_collage.py --src test_caries_full/testset --out reports/collage.png
"""
import argparse
import math
from pathlib import Path
from PIL import Image, ImageOps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="test_caries_full/testset")
    ap.add_argument("--out", default="reports/collage.png")
    ap.add_argument("--tile", type=int, default=400,
                    help="tile side (px). 400 * 20 cols = 8000px wide")
    ap.add_argument("--cols", type=int, default=20)
    ap.add_argument("--bg", default="white")
    ap.add_argument("--pad", type=int, default=4)
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() == ".png")
    print(f"found {len(files)} PNGs in {src}")

    cols = args.cols
    rows = math.ceil(len(files) / cols)
    T = args.tile
    P = args.pad
    W = cols * T + (cols + 1) * P
    H = rows * T + (rows + 1) * P
    print(f"grid {cols}x{rows}, tile {T}px, canvas {W}x{H}")

    canvas = Image.new("RGB", (W, H), args.bg)
    for i, f in enumerate(files):
        r, c = divmod(i, cols)
        img = Image.open(f).convert("RGB")
        # Fit into TxT preserving aspect ratio, center on a TxT background
        img.thumbnail((T, T), Image.LANCZOS)
        tile = Image.new("RGB", (T, T), args.bg)
        tile.paste(img, ((T - img.width) // 2, (T - img.height) // 2))
        x = P + c * (T + P)
        y = P + r * (T + P)
        canvas.paste(tile, (x, y))
        if (i + 1) % 50 == 0:
            print(f"  placed {i+1}/{len(files)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    print(f"wrote {out} ({W}x{H})")


if __name__ == "__main__":
    main()
