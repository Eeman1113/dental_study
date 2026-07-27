#!/usr/bin/env python3
"""
two_stage.py - YOLO locates, MedGemma describes.

Stage 1: run the fine-tuned YOLO detector to get bounding boxes + class labels.
Stage 2: crop each detection (with padding), pad to square, ask MedGemma via
Ollama for a description of what's in the crop.

The crop-and-ask design fixes the biggest failure mode of the whole-image
approach (main.py): MedGemma can rationalize a box it drew in a region it
never actually looked at closely. Here it only sees the crop, with no image
context to invent from - the description either matches the crop or it doesn't.

    python two_stage.py tooth.png
    python two_stage.py tooth.png --conf 0.4 --pad 0.2 --save-crops --out annotated.png

Requires: pillow, ultralytics; Ollama running with medgemma1.5 pulled.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

# reuse the helpers already in main.py
from main import pad_to_square, to_b64, query_ollama, split_thinking

CROP_PROMPT = (
    "This is a close-up crop taken from an intraoral photograph. Describe "
    "exactly what you see. Is there evidence of dental caries (tooth decay) - "
    "for example a cavitated lesion, a dark shadow inside enamel, or a "
    "brown/black discolouration on a tooth surface? Answer in two short "
    "sentences: first what you see, then whether it looks like caries and "
    "how confident you are."
)

CLASS_COLOR = {"d": (60, 200, 60), "D": (240, 130, 40)}


def pad_box(px0, py0, px1, py1, W, H, pad_frac):
    bw, bh = px1 - px0, py1 - py0
    px0 = max(0, int(px0 - bw * pad_frac))
    py0 = max(0, int(py0 - bh * pad_frac))
    px1 = min(W, int(px1 + bw * pad_frac))
    py1 = min(H, int(py1 + bh * pad_frac))
    return px0, py0, px1, py1


def annotate(img, dets, verdicts):
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(18, out.width // 60))
    except Exception:
        font = ImageFont.load_default()
    lw = max(3, out.width // 400)
    for (cls, conf, box), verdict in zip(dets, verdicts):
        color = CLASS_COLOR.get(cls, (255, 40, 40))
        px0, py0, px1, py1 = box
        draw.rectangle([px0, py0, px1, py1], outline=color, width=lw)
        tag = f"{cls} {conf:.2f}"
        draw.text((px0 + 6, max(py0 - font.size - 4, 2)), tag, fill=color, font=font)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default="/opt/homebrew/runs/detect/caries_yolo/weights/best.pt")
    ap.add_argument("--model", default="medgemma1.5", help="Ollama model tag")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="drop YOLO detections below this confidence")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="pad each crop by this fraction of box side (context helps VLMs)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="two_stage_annotated.png")
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--json", default="two_stage_report.json")
    args = ap.parse_args()

    img_path = Path(args.image)
    original = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    W, H = original.size
    print(f"image: {img_path.name} ({W}x{H})")

    # --- stage 1: YOLO ------------------------------------------------
    yolo = YOLO(args.weights)
    r = yolo.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf,
                     device=args.device, verbose=False)[0]
    dets = []
    for b in r.boxes:
        cls = r.names[int(b.cls.item())]
        conf = float(b.conf.item())
        xyxy = [int(round(v)) for v in b.xyxy[0].tolist()]
        dets.append((cls, conf, xyxy))
    dets.sort(key=lambda x: -x[1])
    print(f"stage 1 (YOLO): {len(dets)} detection(s) at conf>={args.conf}")
    for i, (cls, conf, box) in enumerate(dets):
        print(f"  [{i}] {cls} conf={conf:.3f} xyxy={box}")
    if not dets:
        annotate(original, [], []).save(args.out)
        Path(args.json).write_text(json.dumps({"image": str(img_path), "detections": []}, indent=2))
        print("no boxes to describe. wrote empty report.")
        return

    # --- stage 2: MedGemma on each crop -------------------------------
    print(f"\nstage 2 (MedGemma via Ollama, model={args.model}):")
    verdicts = []
    report = {"image": str(img_path.resolve()), "detections": []}
    for i, (cls, conf, box) in enumerate(dets):
        pbox = pad_box(*box, W, H, args.pad)
        crop = original.crop(pbox)
        if args.save_crops:
            cp = Path(f"two_stage_crop_{i}.png")
            crop.save(cp)
        padded, _, _, _ = pad_to_square(crop)
        thinking, answer = split_thinking(
            query_ollama(args.model, CROP_PROMPT, to_b64(padded), args.host))
        verdicts.append(answer.strip())
        print(f"\n  [{i}] YOLO says {cls!r} ({conf:.2f}) at {box}")
        if thinking:
            print(f"      trace: {thinking[:200]}{'...' if len(thinking) > 200 else ''}")
        print(f"      MedGemma: {answer.strip()}")
        report["detections"].append({
            "index": i,
            "yolo_class": cls,
            "yolo_confidence": round(conf, 4),
            "box_xyxy": box,
            "padded_box_xyxy": list(pbox),
            "medgemma_thinking": thinking,
            "medgemma_answer": answer.strip(),
        })

    annotate(original, dets, verdicts).save(args.out)
    Path(args.json).write_text(json.dumps(report, indent=2))
    print(f"\nannotated -> {args.out}")
    print(f"report    -> {args.json}")


if __name__ == "__main__":
    main()
