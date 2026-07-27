#!/usr/bin/env python3
"""
prepare_data.py - inspect the caries dataset and build ONE patient-aware split
that both YOLO and MedGemma will reuse.

Why this exists (read before running):
  * If the same patient's mouth appears in both train and test, every metric
    you produce is inflated and you won't discover it until deployment. This
    script tries to group images by patient from the filename and splits at the
    GROUP level. You MUST eyeball the reported grouping - if it says
    "groups == images", grouping failed and the split is image-level (warned).
  * It reports class balance and box-size distribution. Caries boxes are tiny;
    that stat, not any hyperparameter, dictates your input resolution.

The Zenodo record ships YOLO/COCO/VOC/LabelMe. We use the YOLO .txt labels.
Point --root at the extracted 'Dataset.zip' folder. The script auto-discovers
image/label locations rather than assuming a fixed tree, because the record's
internal layout isn't documented on the landing page.

    python prepare_data.py --root ./Dataset --out ./yolo_split
    # then inspect the printout, adjust --patient-regex if grouping looks wrong

Requires: pillow, scikit-learn, pyyaml, numpy
"""

import argparse
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_pairs(root: Path):
    """Find (image, label.txt) pairs. Matches by identical stem anywhere under root."""
    images, labels = {}, {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            images.setdefault(p.stem, p)
        elif p.suffix.lower() == ".txt" and p.name.lower() not in {"classes.txt", "readme.txt"}:
            labels.setdefault(p.stem, p)
    pairs = [(images[s], labels[s]) for s in images if s in labels]
    orphan_imgs = [images[s] for s in images if s not in labels]
    print(f"found {len(images)} images, {len(labels)} label files")
    print(f"  -> {len(pairs)} matched pairs, {len(orphan_imgs)} images with no label")
    if orphan_imgs:
        # Images with no .txt are treated as NEGATIVES (no lesion). Keep them -
        # they are what stops the model from always finding something. But make
        # sure they're genuinely negative and not just unlabeled/missing.
        print(f"  NOTE: {len(orphan_imgs)} unlabeled images will be kept as negatives.")
        print(f"        Verify these are truly lesion-free, not just un-annotated.")
        for img in orphan_imgs:
            pairs.append((img, None))
    return pairs


def read_class_names(root: Path, n_fallback: int):
    for name in ("classes.txt", "data.yaml", "obj.names"):
        for hit in root.rglob(name):
            if hit.suffix == ".yaml":
                try:
                    d = yaml.safe_load(hit.read_text())
                    if isinstance(d.get("names"), (list, dict)):
                        vals = d["names"]
                        return list(vals.values()) if isinstance(vals, dict) else vals
                except Exception:
                    pass
            else:
                lines = [l.strip() for l in hit.read_text().splitlines() if l.strip()]
                if lines:
                    print(f"read class names from {hit}")
                    return lines
    print("no class-name file found; using placeholder names - EDIT data.yaml after.")
    return [f"class_{i}" for i in range(n_fallback)]


def parse_label(lbl: Path):
    """Return list of (cls, cx, cy, w, h) from a YOLO txt, or [] for negatives."""
    if lbl is None:
        return []
    out = []
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            c, cx, cy, w, h = parts[:5]
            out.append((int(float(c)), float(cx), float(cy), float(w), float(h)))
    return out


def patient_id(stem: str, pattern: str):
    """Extract a patient/group key from the filename stem."""
    if pattern:
        m = re.search(pattern, stem)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    # default heuristic: leading token before first separator or digit-run
    m = re.match(r"([A-Za-z]*\d+|[A-Za-z]+)", stem)
    return m.group(0) if m else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted Dataset folder")
    ap.add_argument("--out", default="./yolo_split")
    ap.add_argument("--patient-regex", default="",
                    help="regex with one capture group for the patient id; "
                         "leave empty to use the default heuristic")
    ap.add_argument("--views", default="",
                    help="comma-separated substrings; keep only images whose "
                         "path contains one of them (e.g. 'Mandibular,Maxillary_Occlusal'). "
                         "Empty = keep all.")
    ap.add_argument("--ratios", default="0.7,0.15,0.15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (use on filesystems "
                         "without symlink support)")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    root, out = Path(args.root), Path(args.out)
    pairs = find_pairs(root)
    if not pairs:
        raise SystemExit("no image/label pairs found - check --root")

    if args.views:
        keep = [v.strip() for v in args.views.split(",") if v.strip()]
        before = len(pairs)
        pairs = [(i, l) for (i, l) in pairs if any(k in str(i) for k in keep)]
        print(f"view filter {keep}: {before} -> {len(pairs)} pairs")
        if not pairs:
            raise SystemExit("view filter removed everything - check --views")

    # ---- stats -------------------------------------------------------
    cls_counter = Counter()
    areas_px, areas_norm = [], []
    boxes_per_img = []
    n_negative = 0
    for img, lbl in pairs:
        boxes = parse_label(lbl)
        boxes_per_img.append(len(boxes))
        if not boxes:
            n_negative += 1
            continue
        try:
            W, H = Image.open(img).size
        except Exception:
            W = H = None
        for c, cx, cy, w, h in boxes:
            cls_counter[c] += 1
            areas_norm.append(w * h)
            if W:
                areas_px.append((w * W) * (h * H))

    n_classes = (max(cls_counter) + 1) if cls_counter else 1
    names = read_class_names(root, n_classes)

    print("\n=== dataset stats ===")
    print(f"images: {len(pairs)}  (negatives / no-lesion: {n_negative})")
    print(f"total boxes: {sum(cls_counter.values())}")
    print("class distribution:")
    for c in sorted(cls_counter):
        nm = names[c] if c < len(names) else f"class_{c}"
        print(f"  {c} ({nm}): {cls_counter[c]}")
    if areas_norm:
        an = np.array(areas_norm)
        print(f"box area (fraction of image): "
              f"p10={np.percentile(an,10)*100:.2f}%  median={np.median(an)*100:.2f}%  "
              f"p90={np.percentile(an,90)*100:.2f}%")
    if areas_px:
        ap_ = np.sqrt(np.array(areas_px))  # side length in px of an equiv square
        print(f"lesion size (equiv. side, px): "
              f"median={np.median(ap_):.0f}  p10={np.percentile(ap_,10):.0f}")
        print("  -> if median side is small vs your imgsz, lesions get lost. "
              "Raise imgsz or tile.")

    # ---- grouping ----------------------------------------------------
    groups = defaultdict(list)
    for i, (img, _) in enumerate(pairs):
        groups[patient_id(img.stem, args.patient_regex)].append(i)
    print(f"\ngrouping: {len(groups)} groups over {len(pairs)} images "
          f"(avg {len(pairs)/len(groups):.1f} imgs/group)")
    sample = list(groups.items())[:5]
    print("  sample groups:", [(k, len(v)) for k, v in sample])
    grouped = len(groups) < 0.9 * len(pairs)
    if not grouped:
        print("  !! grouping looks image-level (nearly one group per image).")
        print("     Your split will NOT be patient-safe. Fix --patient-regex if")
        print("     filenames actually encode a patient id.")

    # ---- split (at group level) --------------------------------------
    r_tr, r_va, r_te = (float(x) for x in args.ratios.split(","))
    gkeys = list(groups.keys())
    random.shuffle(gkeys)
    n = len(gkeys)
    tr_cut, va_cut = int(n * r_tr), int(n * (r_tr + r_va))
    split_of = {}
    for gi, k in enumerate(gkeys):
        s = "train" if gi < tr_cut else ("val" if gi < va_cut else "test")
        for idx in groups[k]:
            split_of[idx] = s

    # ---- materialize YOLO layout -------------------------------------
    for sub in ("train", "val", "test"):
        (out / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sub).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    link = shutil.copy2 if args.copy else os.symlink
    for i, (img, lbl) in enumerate(pairs):
        s = split_of[i]
        counts[s] += 1
        dst_img = out / "images" / s / img.name
        dst_lbl = out / "labels" / s / (img.stem + ".txt")
        for src, dst in ((img.resolve(), dst_img), ):
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            link(src, dst)
        # negatives get an empty label file (valid in YOLO = background image)
        if lbl is None:
            dst_lbl.write_text("")
        else:
            if dst_lbl.exists():
                dst_lbl.unlink()
            link(lbl.resolve(), dst_lbl)

    print(f"\nsplit counts: {dict(counts)}")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"wrote {out/'data.yaml'}")
    if names[0].startswith("class_"):
        print("REMEMBER: edit the 'names' in data.yaml to the real class labels.")


if __name__ == "__main__":
    main()
