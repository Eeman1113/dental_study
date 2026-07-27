#!/usr/bin/env python3
"""
test_full.py - full-scale evaluation of best.pt + medgemma1.5 on the test split.

For every image in yolo_split/images/test:
  - runs the detector
  - runs medgemma on each detection crop
  - matches detections to ground truth by IoU >= 0.5 (TP/FP)
  - writes a client-facing PNG (via analyze.build_panels) to test/testset/
  - writes a per-image JSON with detections + reasoning + TP/FP labels

At the end, aggregates:
  - YOLO-only metrics (P/R/mAP-like) on the test set
  - medgemma verdict cross-tab (confirm/uncertain/reject vs TP/FP)
  - combined-pipeline metrics if we drop medgemma-rejected detections
  - writes test/METRICS.json and test/REPORT.md

Uses eval_two_stage.py's parser and match logic; uses analyze.py's build_panels.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from ultralytics import YOLO

from main import pad_to_square, to_b64, query_ollama, split_thinking
from analyze import build_panels, CROP_PROMPT
from two_stage import pad_box
from eval_two_stage import parse_verdict, iou, yolo_to_xyxy, match_predictions


def format_dt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="./yolo_split")
    ap.add_argument("--weights", default="./best.pt")
    ap.add_argument("--out", default="./test")
    ap.add_argument("--model", default="medgemma1.5")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="cap number of images (0 = all)")
    args = ap.parse_args()

    split = Path(args.split)
    img_dir = split / "images" / "test"
    lbl_dir = split / "labels" / "test"
    out = Path(args.out)
    (out / "testset").mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        images = images[:args.limit]
    print(f"test images to process: {len(images)}")

    # --- Stage A: YOLO-only test eval (fast, gives us mAP metrics) ----
    print("\n=== stage A: detector-only test-set metrics ===")
    detector = YOLO(args.weights)
    yolo_metrics = detector.val(data=str(split / "data.yaml"), split="test",
                                imgsz=args.imgsz, device=args.device, verbose=False)
    yolo_stats = {
        "mAP50":    float(yolo_metrics.box.map50),
        "mAP50-95": float(yolo_metrics.box.map),
        "precision": float(yolo_metrics.box.mp),
        "recall":    float(yolo_metrics.box.mr),
        "per_class": {},
    }
    for i, name in yolo_metrics.names.items():
        if i < len(yolo_metrics.box.maps):
            yolo_stats["per_class"][name] = {
                "mAP50":    float(yolo_metrics.box.ap50[i]),
                "mAP50-95": float(yolo_metrics.box.maps[i]),
                "precision": float(yolo_metrics.box.p[i]),
                "recall":    float(yolo_metrics.box.r[i]),
            }
    print(f"  mAP50={yolo_stats['mAP50']:.4f}  mAP50-95={yolo_stats['mAP50-95']:.4f}"
          f"  P={yolo_stats['precision']:.4f}  R={yolo_stats['recall']:.4f}")

    # --- Stage B: per-image combined pipeline + rendering -------------
    print(f"\n=== stage B: combined pipeline on {len(images)} images ===")
    per_det = []
    per_img = []
    t0 = time.time()

    for idx, img_path in enumerate(images):
        original = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        W, H = original.size

        # ground truth
        gt_path = lbl_dir / (img_path.stem + ".txt")
        gts = []
        if gt_path.exists() and gt_path.stat().st_size > 0:
            for line in gt_path.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    gts.append(yolo_to_xyxy(int(float(p[0])), float(p[1]),
                                            float(p[2]), float(p[3]), float(p[4]), W, H))

        # detector inference
        r = detector.predict(source=str(img_path), imgsz=args.imgsz,
                             conf=args.conf, device=args.device, verbose=False)[0]
        preds = []
        for b in r.boxes:
            preds.append({
                "cls": r.names[int(b.cls.item())],
                "conf": float(b.conf.item()),
                "box": tuple(int(round(v)) for v in b.xyxy[0].tolist()),
            })
        preds.sort(key=lambda x: -x["conf"])

        # medgemma per crop
        detections = []
        for i, p in enumerate(preds):
            pbox = pad_box(*p["box"], W, H, args.pad)
            crop = original.crop(pbox)
            padded, _, _, _ = pad_to_square(crop)
            _, answer = split_thinking(
                query_ollama(args.model, CROP_PROMPT, to_b64(padded), args.host))
            answer = " ".join(answer.split())
            detections.append({
                "index": i,
                "cls": p["cls"],
                "confidence": p["conf"],
                "box": list(p["box"]),
                "padded_box": list(pbox),
                "reasoning": answer,
                "mg_verdict": parse_verdict(answer),
            })

        # IoU match to GT
        matched, n_fn = match_predictions(
            [{"conf": d["confidence"], "box": d["box"]} for d in detections],
            gts, args.iou_thr,
        )
        for di, det in enumerate(detections):
            tp_fp, gt_idx, iouv = matched[di]
            det["tp_fp"] = tp_fp
            det["gt_index"] = gt_idx
            det["iou_with_gt"] = round(iouv, 4)

        # per-image aggregates
        n_tp = sum(1 for m in matched if m[0] == "TP")
        n_fp = sum(1 for m in matched if m[0] == "FP")
        per_img.append({
            "img": img_path.name, "n_gt": len(gts), "n_pred": len(detections),
            "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn,
        })
        for d in detections:
            per_det.append({
                "img": img_path.name, "yolo_class": d["cls"],
                "yolo_conf": round(d["confidence"], 4),
                "tp_fp": d["tp_fp"], "iou": d["iou_with_gt"],
                "mg_verdict": d["mg_verdict"], "mg_answer": d["reasoning"],
            })

        # render + save
        out_png = out / "testset" / (img_path.stem + ".png")
        build_panels(original, detections).save(out_png)
        out_json = out / "testset" / (img_path.stem + ".json")
        out_json.write_text(json.dumps({
            "image": str(img_path.resolve()),
            "gt_boxes_xyxy": gts,
            "detections": detections,
            "matched": {"tp": n_tp, "fp": n_fp, "fn": n_fn},
        }, indent=2))

        elapsed = time.time() - t0
        eta = (len(images) - idx - 1) * elapsed / max(1, idx + 1)
        print(f"[{idx+1:>3}/{len(images)}] {img_path.name:<70} "
              f"gt={len(gts)} pred={len(detections)} "
              f"tp={n_tp} fp={n_fp} fn={n_fn} "
              f"| {format_dt(elapsed)} elapsed, eta {format_dt(eta)}",
              flush=True)

    # --- Stage C: aggregate combined-pipeline metrics -----------------
    print("\n=== stage C: aggregating combined-pipeline metrics ===")
    tp = sum(1 for r in per_det if r["tp_fp"] == "TP")
    fp = sum(1 for r in per_det if r["tp_fp"] == "FP")
    fn = sum(im["n_fn"] for im in per_img)
    P  = tp / (tp + fp) if (tp + fp) else 0
    R  = tp / (tp + fn) if (tp + fn) else 0

    ct = Counter()
    for r in per_det:
        ct[(r["mg_verdict"], r["tp_fp"])] += 1
    fp_rej = ct[("reject", "FP")]
    tp_rej = ct[("reject", "TP")]
    fp_kept = ct[("confirm", "FP")] + ct[("uncertain", "FP")]
    tp_kept = ct[("confirm", "TP")] + ct[("uncertain", "TP")]
    new_P = tp_kept / (tp_kept + fp_kept) if (tp_kept + fp_kept) else 0
    new_R = tp_kept / (tp_kept + fn + tp_rej) if (tp_kept + fn + tp_rej) else 0

    metrics = {
        "test_images": len(images),
        "detector_only_ultralytics_val": yolo_stats,
        "combined_pipeline_at_conf_thr": {
            "conf_thr": args.conf, "iou_thr": args.iou_thr,
            "total_detections": len(per_det), "total_gt": tp + fn,
            "yolo_alone": {"TP": tp, "FP": fp, "FN": fn, "P": P, "R": R},
            "cross_tab": {f"{v}/{k}": n for (v, k), n in ct.items()},
            "yolo_medgemma_veto": {
                "TP": tp_kept, "FP": fp_kept, "FN": fn + tp_rej,
                "P": new_P, "R": new_R,
                "delta_P_vs_yolo_alone": new_P - P,
                "delta_R_vs_yolo_alone": new_R - R,
            },
            "fp_removed_by_veto_fraction": fp_rej / max(fp, 1),
            "tp_lost_by_veto_fraction":    tp_rej / max(tp, 1),
        },
    }
    (out / "METRICS.json").write_text(json.dumps(metrics, indent=2))
    print(f"  wrote {out}/METRICS.json")

    # --- Stage D: human-readable report -------------------------------
    md = []
    md.append("# Full test-set evaluation\n")
    md.append(f"- test images: **{len(images)}**\n")
    md.append(f"- detector: `{args.weights}` at imgsz={args.imgsz}, conf>={args.conf}\n")
    md.append(f"- second-stage model: `{args.model}` via Ollama\n")
    md.append(f"- IoU threshold for TP: {args.iou_thr}\n\n")
    md.append("## Detector-only (ultralytics val)\n\n")
    md.append("| metric | value |\n|---|---|\n")
    md.append(f"| mAP50 | {yolo_stats['mAP50']:.4f} |\n")
    md.append(f"| mAP50-95 | {yolo_stats['mAP50-95']:.4f} |\n")
    md.append(f"| Precision | {yolo_stats['precision']:.4f} |\n")
    md.append(f"| Recall | {yolo_stats['recall']:.4f} |\n\n")
    md.append("### Per class\n\n| class | mAP50 | mAP50-95 | P | R |\n|---|---|---|---|---|\n")
    for name, m in yolo_stats["per_class"].items():
        md.append(f"| {name} | {m['mAP50']:.4f} | {m['mAP50-95']:.4f} | "
                  f"{m['precision']:.4f} | {m['recall']:.4f} |\n")
    md.append(f"\n**Paper's benchmark (image-level split, has patient leakage):** "
              f"YOLOv8s mAP50=0.841, P=0.828, R=0.807.\n\n"
              f"Our numbers are on a patient-safe occlusal-only split -- harder, "
              f"and closer to real deployment.\n\n")
    md.append("## Combined pipeline (detector + medgemma)\n\n")
    md.append(f"total detections at conf>={args.conf}: **{len(per_det)}**\n\n")
    md.append("### YOLO alone (this threshold)\n\n")
    md.append(f"TP={tp}  FP={fp}  FN={fn}  P={P:.4f}  R={R:.4f}\n\n")
    md.append("### MedGemma verdict vs actual label\n\n")
    md.append("| verdict | TP | FP |\n|---|---|---|\n")
    for v in ("confirm", "uncertain", "reject"):
        md.append(f"| {v} | {ct[(v,'TP')]} | {ct[(v,'FP')]} |\n")
    md.append(f"\n### YOLO + MedGemma veto (drop 'reject' verdicts)\n\n")
    md.append(f"TP={tp_kept}  FP={fp_kept}  FN={fn+tp_rej}  P={new_P:.4f}  R={new_R:.4f}\n\n")
    md.append(f"delta vs YOLO alone: P **{new_P-P:+.4f}**, R **{new_R-R:+.4f}**.\n\n")
    md.append(f"- FPs removed by veto: {fp_rej}/{fp} = {fp_rej/max(fp,1):.1%}\n")
    md.append(f"- TPs lost by veto:    {tp_rej}/{tp} = {tp_rej/max(tp,1):.1%}\n\n")
    md.append(f"## Per-image analyses\n\n")
    md.append(f"Full client-facing PNGs and structured JSONs are in `test/testset/` "
              f"({len(images)} of each).\n")
    (out / "REPORT.md").write_text("".join(md))
    print(f"  wrote {out}/REPORT.md")
    print(f"\ntotal wall time: {format_dt(time.time() - t0)}")


if __name__ == "__main__":
    main()
