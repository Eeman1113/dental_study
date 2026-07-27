# Full test-set evaluation
- test images: **30**
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

total detections at conf>=0.25: **75**

### YOLO alone (this threshold)

TP=56  FP=19  FN=2  P=0.7467  R=0.9655

### MedGemma verdict vs actual label

| verdict | TP | FP |
|---|---|---|
| confirm | 35 | 13 |
| uncertain | 13 | 2 |
| reject | 8 | 4 |

### YOLO + MedGemma veto (drop 'reject' verdicts)

TP=48  FP=15  FN=10  P=0.7619  R=0.8276

delta vs YOLO alone: P **+0.0152**, R **-0.1379**.

- FPs removed by veto: 4/19 = 21.1%
- TPs lost by veto:    8/56 = 14.3%

## Per-image analyses

Full client-facing PNGs and structured JSONs are in `test/testset/` (30 of each).
