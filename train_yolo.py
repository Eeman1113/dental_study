#!/usr/bin/env python3
"""
train_yolo.py - the baseline detector, tuned for a 16 GB base-M4.

Hardware reality: on a 16 GB unified-memory M4, "GPU memory" is system RAM.
An earlier attempt with yolo11m at imgsz=1280 allocated 15.5 GB and starved
the OS - iterations took 30+ minutes because everything swapped. Defaults
below stay under ~8 GB so macOS keeps a working set and iterations are
actually GPU-bound, not swap-bound.

    # 3-epoch probe (measure real per-iter time before committing)
    python train_yolo.py --data ./yolo_split/data.yaml --epochs 3
    # then the full run
    python train_yolo.py --data ./yolo_split/data.yaml

Requires: ultralytics  (pip install ultralytics)
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml from prepare_data.py")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="yolo11n/s/m/l/x .pt. On 16 GB M4, s is the ceiling; "
                         "m/l/x will thrash swap.")
    ap.add_argument("--imgsz", type=int, default=896,
                    help="input resolution. 896 is a compromise: median lesion "
                         "(266 px in raw) still lands ~185 px after resize.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8,
                    help="batch=8 at imgsz=896 with yolo11s fits ~7 GB on M4.")
    ap.add_argument("--workers", type=int, default=2,
                    help="dataloader workers. M4 has 4 P-cores; 2 leaves 2 for training.")
    ap.add_argument("--device", default="mps", help="mps (Apple), 0 (CUDA), or cpu")
    ap.add_argument("--name", default="caries_yolo")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        name=args.name,
        resume=args.resume,
        patience=30,              # early-stop if val mAP plateaus
        cache="disk",             # cache preprocessed labels to disk, not RAM (RAM is precious on M4)
        amp=False,                # MPS AMP has silent-fallback bugs; disable for reliability
        rect=False,               # rectangular training conflicts with mosaic; keep mosaic
        # --- augmentation tuned for intraoral photos ---
        mosaic=1.0,
        close_mosaic=15,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,   # mild colour jitter - lighting varies
        fliplr=0.5,
        flipud=0.0,               # a mouth is not vertically symmetric; don't flip up/down
        degrees=10.0,
        translate=0.1, scale=0.4,
        # --- loss / schedule ---
        optimizer="SGD",          # explicit — stops ultralytics from ignoring lr0
        cos_lr=True,
        lr0=0.01, lrf=0.01,
        box=7.5, cls=0.5,         # bump cls weight later if per-class mAP shows imbalance
    )

    # Evaluate on the HELD-OUT test split, not val. This is the honest number.
    print("\n=== evaluating on test split ===")
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device)
    print(f"test mAP@50    : {metrics.box.map50:.4f}")
    print(f"test mAP@50-95 : {metrics.box.map:.4f}")
    print(f"test precision : {metrics.box.mp:.4f}")
    print(f"test recall    : {metrics.box.mr:.4f}")
    print("\nPer-class recall is the one to watch for caries - a missed lesion")
    print("(low recall) is a worse failure here than a false alarm (low precision).")

    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    print(f"\nbest weights -> {best}")
    print("Use this .pt as the detector in the two-stage pipeline (crop -> MedGemma).")


if __name__ == "__main__":
    main()
