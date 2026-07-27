#!/usr/bin/env python3
"""
analyze.py - clinician-facing analysis of an intraoral photograph.

Produces a two-panel image: the original photograph on the left, the annotated
version on the right, and a findings panel below with per-lesion reasoning.

    python analyze.py path/to/photo.jpg --out ./test/photo_analysis.png
    python analyze.py path/to/photo.jpg --conf 0.30 --pad 0.20
"""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

from main import pad_to_square, to_b64, query_ollama, split_thinking

CROP_PROMPT = (
    "This is a close-up crop taken from an intraoral photograph. Describe "
    "exactly what you see. Is there evidence of dental caries (tooth decay) - "
    "for example a cavitated lesion, a dark shadow inside enamel, or a "
    "brown/black discolouration on a tooth surface? Answer in two short "
    "sentences: first what you see, then whether it looks like caries and "
    "how confident you are."
)

# Per the source paper: 'd' = primary tooth decay, 'D' = permanent tooth decay.
LABEL_LONG  = {"d": "Primary caries",   "D": "Permanent caries"}
LABEL_COLOR = {"d": (60, 200, 60),      "D": (240, 130, 40)}


def pad_box(px0, py0, px1, py1, W, H, pad_frac):
    bw, bh = px1 - px0, py1 - py0
    return (max(0, int(px0 - bw * pad_frac)), max(0, int(py0 - bh * pad_frac)),
            min(W, int(px1 + bw * pad_frac)), min(H, int(py1 + bh * pad_frac)))


# Fixed output geometry — everything below sizes off this, no post-scaling.
OUT_W       = 2400           # total output width (px) — matches the reference layout
GAP         = 32             # gap between the two image panels
SIDE_PAD    = 36             # left/right padding of the findings panel
PANEL_W     = (OUT_W - GAP) // 2   # each photo panel width (~1184)


def load_fonts():
    """Fonts calibrated for the 2400-wide output, sized to stay readable even
    when a preview pane scales the image down to ~1/3.
    Helvetica.ttc is a collection - index 0 = Regular, index 1 = Bold."""
    HELV = "/System/Library/Fonts/Helvetica.ttc"
    try:
        f_title = ImageFont.truetype(HELV, 68, index=1)   # Bold
        f_head  = ImageFont.truetype(HELV, 54, index=1)   # Bold
        f_box   = ImageFont.truetype(HELV, 52, index=1)   # Bold
        f_body  = ImageFont.truetype(HELV, 44, index=0)   # Regular
    except OSError:
        # Last-resort fallback so we don't silently render at size 10.
        f_title = f_head = f_box = f_body = ImageFont.load_default(size=40)
    return f_title, f_head, f_box, f_body


def fit_to_panel(img, target_w):
    """Downscale image to fit target width; return (fitted_img, scale_factor)."""
    if img.width <= target_w:
        return img.copy(), 1.0
    scale = target_w / img.width
    return img.resize((target_w, int(img.height * scale)), Image.LANCZOS), scale


def draw_annotations_scaled(img_scaled, detections, scale, f_box):
    """Draw boxes/labels on the ALREADY-SCALED image, scaling coords with it."""
    out = img_scaled.copy()
    draw = ImageDraw.Draw(out)
    lw = max(3, out.width // 300)
    for det in detections:
        color = LABEL_COLOR.get(det["cls"], (255, 40, 40))
        px0, py0, px1, py1 = [int(round(v * scale)) for v in det["box"]]
        draw.rectangle([px0, py0, px1, py1], outline=color, width=lw)
        tag = f'[{det["index"]}] {LABEL_LONG.get(det["cls"], det["cls"])}'
        tb = draw.textbbox((0, 0), tag, font=f_box)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(py0 - th - 8, 2)
        draw.rectangle([px0, ty, px0 + tw + 12, ty + th + 8], fill=color)
        draw.text((px0 + 6, ty + 2), tag, fill="white", font=f_box)
    return out


def build_findings_panel(detections, width, fonts):
    f_title, f_head, _, f_body = fonts
    line_h  = int(f_body.size * 1.4)
    head_h  = int(f_head.size * 1.5)
    title_h = int(f_title.size * 1.6)
    # Rough conservative wrap column count. Helvetica avg glyph width ~= 0.55 * size.
    avg_char_w = max(8, int(f_body.size * 0.55))
    wrap_cols  = max(40, (width - SIDE_PAD * 2 - 28) // avg_char_w)

    h = SIDE_PAD * 2 + title_h
    wraps = []
    for det in detections:
        w = textwrap.wrap(det["reasoning"], width=wrap_cols, break_long_words=False)
        wraps.append(w)
        h += head_h + line_h * max(1, len(w)) + 14
    if not detections:
        h += head_h

    panel = Image.new("RGB", (width, h), "white")
    pd = ImageDraw.Draw(panel)
    y = SIDE_PAD
    pd.text((SIDE_PAD, y), "Findings", fill="black", font=f_title)
    y += title_h

    if not detections:
        pd.text((SIDE_PAD, y), "No lesions detected in this photograph.",
                fill=(40, 40, 40), font=f_body)
        return panel

    for det, wrapped in zip(detections, wraps):
        color = LABEL_COLOR.get(det["cls"], (255, 40, 40))
        pd.rectangle([SIDE_PAD, y + 8, SIDE_PAD + 14, y + head_h - 6], fill=color)
        header = f'[{det["index"]}] {LABEL_LONG.get(det["cls"], det["cls"])}'
        pd.text((SIDE_PAD + 26, y), header, fill="black", font=f_head)
        y += head_h
        for line in wrapped:
            pd.text((SIDE_PAD + 26, y), line, fill=(40, 40, 40), font=f_body)
            y += line_h
        y += 14
    return panel


def build_panels(original, detections):
    fonts = load_fonts()
    f_title, _, f_box, _ = fonts

    left,  s = fit_to_panel(original, PANEL_W)
    right = draw_annotations_scaled(left, detections, s, f_box)

    lbl_h = int(f_title.size * 1.6)
    photo_h = left.height
    top = Image.new("RGB", (OUT_W, lbl_h + photo_h + 16), "white")
    td = ImageDraw.Draw(top)
    td.text((14, 10), "Original photograph", fill="black", font=f_title)
    td.text((PANEL_W + GAP + 14, 10), "Analysis", fill="black", font=f_title)
    top.paste(left,  (0, lbl_h))
    top.paste(right, (PANEL_W + GAP, lbl_h))

    findings = build_findings_panel(detections, OUT_W, fonts)
    combined = Image.new("RGB", (OUT_W, top.height + findings.height), "white")
    combined.paste(top, (0, 0))
    combined.paste(findings, (0, top.height))
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default="./best.pt")
    ap.add_argument("--model", default="medgemma1.5")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    img_path = Path(args.image)
    original = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    W, H = original.size
    print(f"analyzing: {img_path.name} ({W}x{H})")

    detector = YOLO(args.weights)
    r = detector.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf,
                         device=args.device, verbose=False)[0]
    raw = []
    for b in r.boxes:
        raw.append({
            "cls": r.names[int(b.cls.item())],
            "conf": float(b.conf.item()),
            "box": tuple(int(round(v)) for v in b.xyxy[0].tolist()),
        })
    raw.sort(key=lambda x: -x["conf"])
    print(f"  {len(raw)} region(s) of interest at threshold {args.conf}")

    detections = []
    for i, d in enumerate(raw):
        pbox = pad_box(*d["box"], W, H, args.pad)
        crop = original.crop(pbox)
        padded, _, _, _ = pad_to_square(crop)
        _, answer = split_thinking(
            query_ollama(args.model, CROP_PROMPT, to_b64(padded), args.host))
        answer = " ".join(answer.split())
        detections.append({
            "index": i,
            "cls": d["cls"],
            "confidence": d["conf"],
            "box": d["box"],
            "padded_box": list(pbox),
            "reasoning": answer,
        })
        print(f"  [{i}] {LABEL_LONG.get(d['cls'], d['cls'])}: {answer[:120]}"
              f"{'...' if len(answer) > 120 else ''}")

    combined = build_panels(original, detections)

    out = Path(args.out) if args.out else img_path.with_name(img_path.stem + "_analysis.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out)
    print(f"  wrote {out}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({
            "image": str(img_path.resolve()),
            "detections": detections,
        }, indent=2))
        print(f"  wrote {args.report}")


if __name__ == "__main__":
    main()
