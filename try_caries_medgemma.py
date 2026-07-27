#!/usr/bin/env python3
"""
try_caries_medgemma.py - run the fine-tuned model with the EXACT training prompt.

Uses build_vlm_data.py's INSTRUCTION verbatim so the model sees the phrasing it
was trained on. Anything else is off-distribution.
"""
import argparse
import sys
from PIL import Image, ImageOps

from main import pad_to_square, to_b64, query_ollama, split_thinking, extract_boxes, box_to_pixels, draw_boxes
from build_vlm_data import INSTRUCTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="caries-medgemma")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", default="/tmp/caries_annotated.png")
    args = ap.parse_args()

    original = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    padded, x_off, y_off, side = pad_to_square(original)
    print(f"image {original.size} -> padded {padded.size}")

    print(f"\nquerying {args.model} with training prompt...")
    resp = query_ollama(args.model, INSTRUCTION, to_b64(padded), args.host)
    thinking, answer = split_thinking(resp)

    print("\n--- model response ---")
    print(answer)
    print("--- end ---\n")

    boxes = extract_boxes(answer)
    if not boxes:
        print("no boxes parsed")
        return

    w, h = original.size
    px_boxes, labels = [], []
    for i, item in enumerate(boxes):
        b = item.get("box_2d")
        if not b or len(b) != 4:
            continue
        px = box_to_pixels(b, x_off, y_off, side, w, h)
        area_pct = 100.0 * (px[2] - px[0]) * (px[3] - px[1]) / (w * h)
        print(f"  [{i}] {item.get('label','')}: norm={b} px={px} ({area_pct:.1f}% of frame)")
        px_boxes.append(px)
        labels.append(item.get("label", ""))

    draw_boxes(original, px_boxes, labels).save(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
