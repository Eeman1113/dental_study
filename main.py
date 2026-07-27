#!/usr/bin/env python3
"""
main.py - probe MedGemma (served by Ollama) for bounding boxes, with grounding checks.

Handles the things the Ollama CLI gets wrong for this task:
  1. applies EXIF orientation (PIL does NOT do this automatically, but Preview
     and VS Code DO - so what you see is not always what the model got)
  2. pads the image to a square BEFORE inference, matching the preprocessing
     used in MedGemma's localization training/eval
  3. un-maps the returned [0,1000] coords back onto the ORIGINAL image

Grounding checks:
  --describe   blind description pass BEFORE asking for boxes (does it even
               see a mouth?)
  --explain    crop each returned box out of the original, feed the crop back
               alone, ask what's in it. This is the honest test - the model
               cannot rationalize its way around a crop that shows beard.

Requires: Pillow  (conda install pillow)

Usage:
    python main.py tooth.png "the front teeth" --raw --describe --explain
    python main.py tooth.png "caries" --rotate -36 --explain
"""

import argparse
import base64
import io
import json
import re
import sys
import urllib.error
import urllib.request

from PIL import Image, ImageDraw, ImageOps

# Copied verbatim from MedGemma's localization notebook. Do not "improve" the
# wording - the model was tuned against this exact framing. Only the
# CXR-specific patient-laterality line has been removed.
PROMPT_TEMPLATE = """Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{{"box_2d": [800, 3, 840, 471], "label": "car"}}, {{"box_2d": [400, 22, 600, 73], "label": "dog"}}]``` is a valid output. Now answer to the user query.

Query:
Where is the {object_name}? Don't give a final answer without reasoning. Output the final answer in the format "Final Answer: X" where X is a JSON list of objects. The object needs a "box_2d" and "label" key. Answer:"""

DESCRIBE_PROMPT = (
    "Describe exactly what you see in this image. What anatomical structures "
    "are visible? Be specific and concise."
)

# Deliberately does NOT mention teeth or what we expect - a leading prompt
# would let the model agree with us regardless of what's in the crop.
VERIFY_PROMPT = (
    "What is shown in this image? Name the visible anatomical structures. "
    "If you cannot identify anything clearly, say so. Answer in one sentence."
)


def pad_to_square(img):
    """Centre-pad to a square. Returns (padded, x_offset, y_offset, side)."""
    img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    x_off = (side - w) // 2
    y_off = (side - h) // 2
    padded = Image.new("RGB", (side, side), (0, 0, 0))
    padded.paste(img, (x_off, y_off))
    return padded, x_off, y_off, side


def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def query_ollama(model, prompt, image_b64, host, timeout=600):
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 1024},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))["response"]
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach Ollama at {host} - is it running? ({e})")


def split_thinking(text):
    """Return (thinking, answer).

    MedGemma emits a reasoning trace between <unused94> and <unused95>. The
    original notebook DISCARDS it. We keep it - it is the whole point here.
    Note the trace may not survive GGUF conversion at all, in which case
    thinking comes back empty and any 'reasoning' is inline in the answer.
    """
    if "<unused95>" in text:
        head, tail = text.split("<unused95>", 1)
        return head.replace("<unused94>", "").strip(), tail.lstrip()
    return "", text.replace("<unused94>", "").lstrip()


def extract_boxes(text):
    """Pull the JSON list out. Fenced block first, then any bare list."""
    for pattern in (r"```json\s*(\[.*?\])\s*```", r"(\[\s*\{.*?\}\s*\])"):
        for match in re.findall(pattern, text, re.DOTALL):
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list) and parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    return []


def box_to_pixels(box, x_off, y_off, side, img_w, img_h):
    """[0,1000] padded-square coords -> original-image pixel box, clamped."""
    y0, x0, y1, x1 = box
    scale = side / 1000.0
    px0 = int(round(x0 * scale - x_off))
    py0 = int(round(y0 * scale - y_off))
    px1 = int(round(x1 * scale - x_off))
    py1 = int(round(y1 * scale - y_off))
    px0, px1 = max(0, min(px0, img_w)), max(0, min(px1, img_w))
    py0, py1 = max(0, min(py0, img_h)), max(0, min(py1, img_h))
    return px0, py0, px1, py1


def draw_boxes(original, pixel_boxes, labels):
    out = original.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(3, out.size[0] // 500)
    for (px0, py0, px1, py1), label in zip(pixel_boxes, labels):
        draw.rectangle([px0, py0, px1, py1], outline=(255, 40, 40), width=width)
        if label:
            draw.text((px0 + 6, max(py0 - 14, 2)), str(label), fill=(255, 40, 40))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("object_name", help='e.g. "the front teeth" or "caries"')
    ap.add_argument("--model", default="medgemma1.5")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", default="annotated.png")
    ap.add_argument("--raw", action="store_true", help="print full model response")
    ap.add_argument("--rotate", type=float, default=0,
                    help="degrees counter-clockwise, applied after EXIF")
    ap.add_argument("--describe", action="store_true",
                    help="blind description pass before asking for boxes")
    ap.add_argument("--explain", action="store_true",
                    help="crop each box and feed it back for verification")
    ap.add_argument("--save-crops", action="store_true",
                    help="write each cropped region to crop_N.png")
    args = ap.parse_args()

    original = ImageOps.exif_transpose(Image.open(args.image))
    if args.rotate:
        original = original.rotate(args.rotate, expand=True, fillcolor=(0, 0, 0))
        print(f"rotated {args.rotate} deg -> {original.size}")

    padded, x_off, y_off, side = pad_to_square(original)
    print(f"image {original.size} -> padded {padded.size} (offset {x_off},{y_off})")
    padded_b64 = to_b64(padded)

    # --- pass 1: blind description -------------------------------------
    if args.describe:
        print("\n=== blind description ===")
        thinking, answer = split_thinking(
            query_ollama(args.model, DESCRIBE_PROMPT, padded_b64, args.host))
        if thinking:
            print(f"[trace] {thinking}\n")
        print(answer.strip())
        print("=== end ===\n")

    # --- pass 2: localization ------------------------------------------
    prompt = PROMPT_TEMPLATE.format(object_name=args.object_name)
    print(f"querying {args.model} for '{args.object_name}' ...")
    thinking, answer = split_thinking(
        query_ollama(args.model, prompt, padded_b64, args.host))

    if thinking:
        print("\n--- reasoning trace ---")
        print(thinking)
        print("--- end trace ---")
    else:
        print("(no <unused94>/<unused95> trace - either stripped in GGUF "
              "conversion, or the model answered without one)")

    if args.raw:
        print("\n--- model response ---")
        print(answer)
        print("--- end ---\n")

    boxes = extract_boxes(answer)
    if not boxes:
        print("No parseable bounding boxes. Re-run with --raw.")
        return

    w, h = original.size
    pixel_boxes, labels = [], []
    print(f"\nparsed {len(boxes)} box(es):")
    for i, item in enumerate(boxes):
        box = item.get("box_2d")
        if not box or len(box) != 4:
            continue
        px = box_to_pixels(box, x_off, y_off, side, w, h)
        if px[2] <= px[0] or px[3] <= px[1]:
            print(f"  ! degenerate box skipped: {box}")
            continue
        area_pct = 100.0 * (px[2] - px[0]) * (px[3] - px[1]) / (w * h)
        print(f"  [{i}] {item.get('label','')}: norm={box} px={px} "
              f"({area_pct:.1f}% of frame)")
        if area_pct > 25:
            print(f"       ^ suspiciously large for a specific structure")
        pixel_boxes.append(px)
        labels.append(item.get("label", ""))

    # --- pass 3: crop-back verification --------------------------------
    if args.explain and pixel_boxes:
        print("\n=== crop-back verification ===")
        print("(feeding each cropped region back ALONE - the model cannot see")
        print(" the full image or its own earlier answer)\n")
        for i, px in enumerate(pixel_boxes):
            crop = original.crop(px)
            if args.save_crops:
                crop.save(f"crop_{i}.png")
                print(f"  wrote crop_{i}.png")
            crop_padded, _, _, _ = pad_to_square(crop)
            _, verdict = split_thinking(
                query_ollama(args.model, VERIFY_PROMPT,
                             to_b64(crop_padded), args.host))
            print(f"  [{i}] claimed '{labels[i]}' -> sees: {verdict.strip()}\n")
        print("=== end ===\n")

    annotated = draw_boxes(original, pixel_boxes, labels)
    annotated.save(args.out)
    print(f"drew {len(pixel_boxes)} box(es) -> {args.out}")


if __name__ == "__main__":
    main()