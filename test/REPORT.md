# Full test-set evaluation
- test images: **332**
- detector: `./best.pt` at imgsz=896, conf>=0.25
- second-stage model: `medgemma1.5` via Ollama
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
| confirm | 473 | 63 |
| uncertain | 95 | 16 |
| reject | 195 | 72 |

### YOLO + MedGemma veto (drop 'reject' verdicts)

TP=568  FP=79  FN=282  P=0.8779  R=0.6682

delta vs YOLO alone: P **+0.0431**, R **-0.2294**.

- FPs removed by veto: 72/151 = 47.7%
- TPs lost by veto:    195/763 = 25.6%

## Per-image analyses

Full client-facing PNGs and structured JSONs are in `test/testset/` (332 of each).
