#!/usr/bin/env python3
"""
eval_two_stage.py - measure whether the YOLO + MedGemma combo beats YOLO alone.

For each test image:
  1. YOLO produces raw detections (class + confidence + box).
  2. Each detection is cropped and shown to MedGemma via Ollama.
  3. MedGemma's free-text verdict is parsed into a categorical signal:
        confirm  = phrases like "moderate/high confidence", "clear evidence"
        uncertain= "possible", "difficult to say", "could be"
        reject   = "no clear evidence", "low confidence", "does not appear"
  4. Each detection is matched to ground truth by IoU >= 0.5 (TP or FP).
  5. We cross-tabulate MedGemma signal vs TP/FP, then compute what happens
     if we drop detections MedGemma rejected.

The point: if MedGemma reliably vetoes FPs without killing TPs, the combined
system has higher precision at the same recall than YOLO alone. If it kills
TPs too, the two-stage design isn't worth its inference cost.

    python eval_two_stage.py --n 20 --conf 0.25
    python eval_two_stage.py --n 50 --conf 0.20 --pad 0.20

Requires: pillow, ultralytics; Ollama running with medgemma1.5 pulled.
"""

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO

from main import pad_to_square, to_b64, query_ollama, split_thinking
from two_stage import CROP_PROMPT, pad_box


# --- MedGemma verdict parser --------------------------------------------
# Ordered so REJECT wins if both confirm and reject phrases appear.
REJECT_PHRASES = (
    "no clear evidence", "no evidence", "does not appear",
    "unlikely", "no visible", "no signs of caries", "no obvious",
    "confidence level: low", "confidence: low", "low confidence",
    "not caries", "healthy",
)
CONFIRM_PHRASES = (
    "high confidence", "highly confident", "clear evidence",
    "definite", "definitely", "clearly shows", "clearly a",
    "moderate confidence", "moderately confident", "appears to be caries",
    "is caries", "evidence of dental caries", "evidence of caries",
    "brown/black discolouration", "cavitated lesion",
)
UNCERTAIN_PHRASES = (
    "possible", "possibly", "could be", "might be", "difficult to determine",
    "hard to say", "uncertain", "unclear", "may indicate", "not definitive",
)


def parse_verdict(text: str) -> str:
    """Return 'confirm', 'reject', or 'uncertain'."""
    t = text.lower()
    if any(p in t for p in REJECT_PHRASES):
        return "reject"
    if any(p in t for p in CONFIRM_PHRASES):
        return "confirm"
    if any(p in t for p in UNCERTAIN_PHRASES):
        return "uncertain"
    return "uncertain"


# --- IoU + TP/FP matcher ------------------------------------------------
def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    aa = (ax1 - ax0) * (ay1 - ay0)
    bb = (bx1 - bx0) * (by1 - by0)
    return inter / (aa + bb - inter)


def yolo_to_xyxy(cls, cx, cy, w, h, W, H):
    return (int((cx - w / 2) * W), int((cy - h / 2) * H),
            int((cx + w / 2) * W), int((cy + h / 2) * H))


def match_predictions(preds, gts, iou_thr=0.5):
    """Greedy IoU matching. Returns list of ('TP'|'FP', matched_gt_idx or None)."""
    assigned = set()
    out = []
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["conf"])
    result_by_idx = {}
    for i in order:
        pbox = preds[i]["box"]
        best_iou, best_g = 0, -1
        for gi, gt in enumerate(gts):
            if gi in assigned:
                continue
            v = iou(pbox, gt)
            if v > best_iou:
                best_iou, best_g = v, gi
        if best_iou >= iou_thr and best_g >= 0:
            assigned.add(best_g)
            result_by_idx[i] = ("TP", best_g, best_iou)
        else:
            result_by_idx[i] = ("FP", None, best_iou)
    for i in range(len(preds)):
        out.append(result_by_idx[i])
    return out, len(gts) - len(assigned)  # + count of missed GTs (FN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="./yolo_split")
    ap.add_argument("--weights", default="/opt/homebrew/runs/detect/caries_yolo/weights/best.pt")
    ap.add_argument("--model", default="medgemma1.5")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--n", type=int, default=20, help="number of test images to sample")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval_two_stage_report.json")
    args = ap.parse_args()

    random.seed(args.seed)
    split = Path(args.split)
    img_dir = split / "images" / "test"
    lbl_dir = split / "labels" / "test"
    all_imgs = sorted(p for p in img_dir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    # prefer images WITH annotations so we actually measure TP/FP behaviour
    with_gt = [p for p in all_imgs
               if (lbl_dir / (p.stem + ".txt")).exists()
               and (lbl_dir / (p.stem + ".txt")).stat().st_size > 0]
    random.shuffle(with_gt)
    sample = with_gt[:args.n]
    print(f"test images with GT: {len(with_gt)} / sampled: {len(sample)}")

    yolo = YOLO(args.weights)

    per_det_rows = []          # every YOLO detection, its verdict, and TP/FP
    per_img = []
    t0 = time.time()
    for img_i, img_path in enumerate(sample):
        original = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        W, H = original.size

        # ground-truth boxes
        gts = []
        for line in (lbl_dir / (img_path.stem + ".txt")).read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                gts.append(yolo_to_xyxy(int(float(p[0])), float(p[1]), float(p[2]),
                                        float(p[3]), float(p[4]), W, H))

        # YOLO detections
        r = yolo.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf,
                         device=args.device, verbose=False)[0]
        preds = []
        for b in r.boxes:
            preds.append({
                "cls": r.names[int(b.cls.item())],
                "conf": float(b.conf.item()),
                "box": tuple(int(round(v)) for v in b.xyxy[0].tolist()),
            })

        # MedGemma per crop
        for p in preds:
            pbox = pad_box(*p["box"], W, H, args.pad)
            crop = original.crop(pbox)
            padded, _, _, _ = pad_to_square(crop)
            _, answer = split_thinking(
                query_ollama(args.model, CROP_PROMPT, to_b64(padded), args.host))
            p["mg_answer"] = answer.strip()
            p["mg_verdict"] = parse_verdict(answer)

        # match to GT
        matched, n_fn = match_predictions(preds, gts, args.iou_thr)
        for pi, p in enumerate(preds):
            tp_fp, gt_i, iouv = matched[pi]
            per_det_rows.append({
                "img": img_path.name,
                "yolo_class": p["cls"],
                "yolo_conf": round(p["conf"], 4),
                "tp_fp": tp_fp,
                "iou": round(iouv, 4),
                "mg_verdict": p["mg_verdict"],
                "mg_answer": p["mg_answer"],
            })

        per_img.append({
            "img": img_path.name,
            "n_gt": len(gts),
            "n_pred": len(preds),
            "n_tp": sum(1 for m in matched if m[0] == "TP"),
            "n_fp": sum(1 for m in matched if m[0] == "FP"),
            "n_fn": n_fn,
        })
        elapsed = time.time() - t0
        remaining = (len(sample) - img_i - 1) * (elapsed / (img_i + 1))
        print(f"[{img_i+1}/{len(sample)}] {img_path.name}: "
              f"gt={len(gts)} pred={len(preds)} "
              f"tp={per_img[-1]['n_tp']} fp={per_img[-1]['n_fp']} fn={per_img[-1]['n_fn']} "
              f"| elapsed {elapsed/60:.1f}m eta {remaining/60:.1f}m")

    # --- aggregate ----------------------------------------------------
    print("\n=== YOLO-alone (all detections at conf>=%.2f) ===" % args.conf)
    tp = sum(1 for r in per_det_rows if r["tp_fp"] == "TP")
    fp = sum(1 for r in per_det_rows if r["tp_fp"] == "FP")
    fn = sum(im["n_fn"] for im in per_img)
    p = tp / (tp + fp) if (tp + fp) else 0
    rc = tp / (tp + fn) if (tp + fn) else 0
    print(f"TP={tp}  FP={fp}  FN={fn}  P={p:.3f}  R={rc:.3f}")

    print("\n=== cross-tab: MedGemma verdict vs actual TP/FP ===")
    ct = Counter()
    for r in per_det_rows:
        ct[(r["mg_verdict"], r["tp_fp"])] += 1
    print(f"{'verdict':<10} {'TP':>6} {'FP':>6}  reject_rate_on_FP  keep_rate_on_TP")
    for v in ("confirm", "uncertain", "reject"):
        t_ = ct[(v, 'TP')]
        f_ = ct[(v, 'FP')]
        print(f"{v:<10} {t_:>6} {f_:>6}")
    fp_rejected = ct[("reject", "FP")]
    tp_rejected = ct[("reject", "TP")]
    fp_kept = ct[("confirm", "FP")] + ct[("uncertain", "FP")]
    tp_kept = ct[("confirm", "TP")] + ct[("uncertain", "TP")]
    print(f"\nif we drop MedGemma-rejected detections:")
    print(f"  FPs removed: {fp_rejected}/{fp} = {fp_rejected/max(fp,1):.1%}")
    print(f"  TPs lost   : {tp_rejected}/{tp} = {tp_rejected/max(tp,1):.1%}")
    new_tp, new_fp = tp_kept, fp_kept
    new_p = new_tp / (new_tp + new_fp) if (new_tp + new_fp) else 0
    new_r = new_tp / (new_tp + fn + tp_rejected) if (new_tp + fn + tp_rejected) else 0
    print(f"\n=== YOLO + MedGemma-veto (drop 'reject') ===")
    print(f"TP={new_tp}  FP={new_fp}  FN={fn + tp_rejected}  P={new_p:.3f}  R={new_r:.3f}")
    print(f"delta vs YOLO alone: P {new_p-p:+.3f}   R {new_r-rc:+.3f}")

    Path(args.out).write_text(json.dumps({
        "config": vars(args),
        "per_image": per_img,
        "per_detection": per_det_rows,
        "aggregate": {
            "yolo_alone": {"TP": tp, "FP": fp, "FN": fn, "P": p, "R": rc},
            "yolo_medgemma_veto": {"TP": new_tp, "FP": new_fp, "FN": fn + tp_rejected,
                                    "P": new_p, "R": new_r},
        },
    }, indent=2))
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
