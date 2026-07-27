# Full test-set evaluation
- test images: **332**
- detector: `./best.pt` at imgsz=896, conf>=0.25
- second-stage model: `caries-medgemma` via Ollama
- IoU threshold for TP: 0.5

## Detector-only (ultralytics val)

| metric | value |
|---|---|
| mAP50 | 0.8597 |
| mAP50-95 | 0.6916 |
| Precision | 0.8243 |
| Recall | 0.8445 |

### Per class

| class | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| d | 0.9353 | 0.7372 | 0.8580 | 0.8764 |
| D | 0.7842 | 0.6459 | 0.7905 | 0.8126 |

**Paper's benchmark (image-level split, has patient leakage):** YOLOv8s mAP50=0.841, P=0.828, R=0.807.

Our numbers are on a patient-safe occlusal-only split -- harder, and closer to real deployment.

## Combined pipeline (detector + medgemma)

total detections at conf>=0.25: **914**

### YOLO alone (this threshold)

TP=763  FP=151  FN=87  P=0.8348  R=0.8976

### MedGemma verdict vs actual label

| verdict | TP | FP |
|---|---|---|
| confirm | 472 | 88 |
| uncertain | 127 | 20 |
| reject | 164 | 43 |

### YOLO + MedGemma veto (drop 'reject' verdicts)

TP=599  FP=108  FN=251  P=0.8472  R=0.7047

delta vs YOLO alone: P **+0.0124**, R **-0.1929**.

- FPs removed by veto: 43/151 = 28.5%
- TPs lost by veto:    164/763 = 21.5%

## Per-image analyses

Full client-facing PNGs and structured JSONs are in `test/testset/` (332 of each).
