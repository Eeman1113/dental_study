#!/usr/bin/env python3
"""
build_vlm_data.py - turn the YOLO split into MedGemma instruction data.

READ THIS, it's the crux of your whole "with proper reasoning" request:

  The dataset has boxes and class labels. It has ZERO reasoning text. That text
  has to come from somewhere, and where it comes from decides whether the model
  learns to think or learns to bluff.

  The WRONG way: hand each labelled image to a big VLM and let it write a
  clinical-sounding justification. That model is told the answer in advance, so
  it learns to rationalize whatever label is present - exactly the burn-marks
  failure you already caught (it boxed a finding, then couldn't see the finding
  in its own crop). Synthetic free-text CoT bakes that in and makes it fluent.

  What this script does instead: builds SHORT reasoning traces that are strictly
  DERIVED FROM THE GROUND-TRUTH GEOMETRY - how many lesions, where in the frame,
  how large. Nothing is invented. It's a scaffold, not clinical reasoning.

  The RIGHT way, if you can afford it: have a dentist write even 200-500 real
  rationales. That beats 6,000 templated ones. Use this to bootstrap, then
  replace the traces with clinician text as you get it.

Output: JSONL, one example per line, in the exact [y0,x0,y1,x1]/1000 protocol
your main.py already uses - so inference code doesn't change after fine-tuning.

    python build_vlm_data.py --split ./yolo_split --out ./vlm_data

Requires: pillow, pyyaml
"""

import argparse
import json
import random
from pathlib import Path

import yaml
from PIL import Image, ImageOps

# Mirrors main.py's localization prompt (CXR laterality line removed). Keep it
# identical to what you'll use at inference.
INSTRUCTION = (
    "Instructions:\nThe following user query will require outputting bounding "
    "boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where "
    "(y0, x0) is the top-left corner and (y1, x1) the bottom-right corner, with "
    "x0 < x1 and y0 < y1. Normalize all coordinates to the range [0, 1000]. "
    "Output a single parseable json list of objects, each with a \"box_2d\" and "
    "a \"label\" key.\n\nQuery:\nIdentify any carious lesions (tooth decay) "
    "visible in this intraoral photograph. Reason briefly about what you see, "
    "then give the final answer as \"Final Answer: X\" where X is the JSON list."
)


def region_phrase(cx, cy):
    """IMAGE-relative descriptor. NOT clinical tooth numbering, NOT patient
    laterality - a photo can't establish those. Kept deliberately geometric."""
    vert = "upper" if cy < 0.45 else ("lower" if cy > 0.55 else "mid")
    horiz = ("left" if cx < 0.4 else ("right" if cx > 0.6 else "central"))
    return f"{vert} {horiz} region of the image"


def to_1000(cx, cy, w, h):
    x0 = max(0, min(1000, round((cx - w / 2) * 1000)))
    y0 = max(0, min(1000, round((cy - h / 2) * 1000)))
    x1 = max(0, min(1000, round((cx + w / 2) * 1000)))
    y1 = max(0, min(1000, round((cy + h / 2) * 1000)))
    return [y0, x0, y1, x1]


# Single-character dataset labels ("d", "D") are poor LM training targets.
# Map to phrasing the VLM can actually reason about.
LABEL_MAP = {"d": "primary_caries", "D": "permanent_caries"}


def build_response(boxes, names):
    """Grounded trace + JSON answer. Every clause traces to a real annotation."""
    if not boxes:
        # NEGATIVE example. These are non-negotiable - without them the model
        # has no "nothing here" path and will hallucinate a lesion every time,
        # which is precisely the behaviour you want to train OUT.
        trace = ("Scanning the visible dentition surface by surface. The enamel "
                 "appears intact with no discoloration, cavitation, or shadowing "
                 "that would indicate decay.")
        return f"{trace}\nFinal Answer: []"

    n = len(boxes)
    locs = []
    objs = []
    for (c, cx, cy, w, h) in boxes:
        raw = names[c] if c < len(names) else f"class_{c}"
        label = LABEL_MAP.get(raw, raw)
        locs.append(f"a {label.replace('_', ' ')} lesion in the {region_phrase(cx, cy)}")
        objs.append({"box_2d": to_1000(cx, cy, w, h), "label": label})
    lead = "one region" if n == 1 else f"{n} regions"
    trace = (f"Scanning the visible dentition. I can identify {lead} of concern: "
             + "; ".join(locs) + ". Marking the affected area"
             + ("." if n == 1 else "s."))
    answer = json.dumps(objs)
    return f"{trace}\nFinal Answer: ```json{answer}```"


def parse_label(lbl: Path):
    if lbl is None or not lbl.exists():
        return []
    out = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append((int(float(p[0])), float(p[1]), float(p[2]),
                        float(p[3]), float(p[4])))
    return out


def build_split(split_dir: Path, names, sub: str, out_path: Path, keep_size: int):
    img_dir = split_dir / "images" / sub
    lbl_dir = split_dir / "labels" / sub
    n_written = n_neg = 0
    with out_path.open("w") as f:
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            boxes = parse_label(lbl_dir / (img.stem + ".txt"))
            if not boxes:
                n_neg += 1
            # optionally normalise orientation + downscale a copy for training
            im = ImageOps.exif_transpose(Image.open(img)).convert("RGB")
            if keep_size and max(im.size) > keep_size:
                im.thumbnail((keep_size, keep_size))
            saved = out_path.parent / "images" / sub / (img.stem + ".png")
            saved.parent.mkdir(parents=True, exist_ok=True)
            im.save(saved)

            record = {
                "image": str(saved.resolve()),
                "messages": [
                    {"role": "user", "content": INSTRUCTION},
                    {"role": "assistant", "content": build_response(boxes, names)},
                ],
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1
    print(f"{sub}: {n_written} examples ({n_neg} negatives) -> {out_path}")
    if n_neg == 0 and sub == "train":
        print("  !! ZERO negatives in train. The model will learn to always find")
        print("     something. Source lesion-free intraoral images before training.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="yolo_split dir from prepare_data.py")
    ap.add_argument("--out", default="./vlm_data")
    ap.add_argument("--keep-size", type=int, default=1024,
                    help="downscale longest side to this for training copies; 0 = keep original")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    split_dir = Path(args.split)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names_map = yaml.safe_load((split_dir / "data.yaml").read_text())["names"]
    names = [names_map[i] for i in sorted(names_map)]
    print(f"classes: {names}")

    for sub in ("train", "val", "test"):
        build_split(split_dir, names, sub, out / f"{sub}.jsonl", args.keep_size)

    print("\nSanity-check a few assistant messages by eye before training.")
    print("If the templated traces read as fake to you, they'll read as fake to")
    print("the model too - that's your cue to get real clinician rationales.")


if __name__ == "__main__":
    main()
