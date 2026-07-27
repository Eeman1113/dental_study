# Daath — YOLO + fine-tuned MedGemma for caries detection

> A two-stage detector for dental caries on intraoral photographs. YOLO locates candidate lesions; MedGemma (LoRA fine-tuned on the same dataset, then GGUF-quantized for Ollama) writes a per-crop verdict and rationale for each detection.

![Pipeline overview](reports/charts/pipeline.png)

Ships with:
- Full source for training, inference, and evaluation
- YOLO11m detector weights (`best.pt`) trained on a **patient-safe, occlusal-only** split — no patient-level leakage between train/val/test
- LoRA adapter for `google/medgemma-1.5-4b-it` (`Dataset/model/medgemma_caries_lora/`)
- GGUF-quantized fine-tune ready for Ollama, hosted on Hugging Face
- End-to-end evaluation on 332 held-out test images with all rendered per-image panels
- A 44MB collage of every single test-set panel

---

## Table of contents

- [Headline result](#headline-result)
- [What the fine-tune actually does](#what-the-fine-tune-actually-does)
- [Sample outputs](#sample-outputs) — 8 close-ups
- [Full test set at a glance](#full-test-set-at-a-glance) — 332-panel collage
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Run inference](#run-inference)
- [Reproduce the evaluation](#reproduce-the-evaluation)
- [Training](#training)
- [Data source](#data-source)
- [License](#license)

---

## Headline result

**Full held-out test set: 332 images · 850 ground-truth lesions · IoU ≥ 0.5 for TP · YOLO confidence ≥ 0.25.**

![Precision / Recall / F1 across pipelines](reports/charts/pr_comparison.png)

| Pipeline | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **YOLO alone** | 763 | 151 | 87 | **0.835** | **0.898** | **0.865** |
| YOLO + base MedGemma veto | 568 | 79 | 282 | 0.878 | 0.668 | 0.759 |
| YOLO + **fine-tuned** MedGemma veto | 599 | 108 | 251 | 0.847 | 0.705 | 0.769 |

**Detector-only mAP:** mAP50 = **0.860**, mAP50-95 = **0.692**.

For context: the source paper reports YOLOv8s mAP50 = 0.841 on an image-level split (which leaks patients between train and test). Our split is at the patient level — strictly harder, closer to real deployment.

![Per-class YOLO detector performance](reports/charts/per_class.png)

---

## What the fine-tune actually does

The LoRA transferred, but not into a useful veto. Two lenses:

### 1) The trade-off — FPs saved vs TPs killed

![FPs saved vs TPs killed](reports/charts/veto_tradeoff.png)

For a veto to help, it needs to remove far more false positives than it kills true positives. **Neither MedGemma configuration achieves that here.**
- Base MedGemma: kills **25.6 % of TPs** to save 47.7 % of FPs.
- Fine-tuned: kills **21.5 % of TPs** to save only 28.5 % of FPs.

The fine-tune is *more permissive* than base (fewer rejects overall), which is why it preserves more TPs — but it also lets more FPs through.

### 2) The verdict × outcome cross-tab

![Fine-tuned MedGemma verdict heatmap](reports/charts/verdict_heatmap.png)

Read the reject row: only **21 % of the fine-tune's "reject" verdicts were actually false positives** — the other 79 % were real lesions the model wrongly killed. That's why veto costs so much recall.

### 3) Whole-image detection collapsed

Fed a full intraoral photo with the training prompt, the fine-tune emits the exact templated negative-example response (`"Scanning the visible dentition surface by surface. The enamel appears intact..."`) across every image, every temperature, every seed. Almost certainly a training-data imbalance — too many all-negatives in the JSONL, so the model memorized the negative template.

**Bottom line:** ship YOLO alone with MedGemma as a *reasoning annotator per detection*, not as a filter. `analyze.py` produces exactly that panel — box + short rationale — leaving the veto decision to a human.

---

## Sample outputs

Each panel below is a full test-image analysis: original photo (left), detector overlay (right), per-detection reasoning below. Filenames encode the failure mode being illustrated.

### The good — YOLO right, MedGemma agrees

**`01_best_confirm`** — 7 detections, 7 correct, MedGemma confirmed all 7.
![Best case](reports/samples/01_best_confirm.png)

**`08_single_clean_tp`** — one lesion, one detection, confidently confirmed.
![Single clean TP](reports/samples/08_single_clean_tp.png)

### The right kind of skepticism — veto correctly kills FPs

**`02_veto_saved_fp`** — YOLO overdetected (2 real + 5 FPs). The fine-tune correctly rejected 3 of the 5 FPs.
![Veto saves FPs](reports/samples/02_veto_saved_fp.png)

### The wrong kind of skepticism — veto kills real lesions

**`03_veto_killed_tp`** — 9 real lesions, all 9 detected, but the fine-tune rejected 2 of them. This is the failure mode that dominates our recall drop.
![Veto kills TPs](reports/samples/03_veto_killed_tp.png)

### The model is fooled — confirms an FP

**`04_confirmed_fp`** — YOLO produced 3 real + 5 FPs. MedGemma confirmed *all 8*, including every false positive. Fluent rationalization of noise.
![Confirmed FPs](reports/samples/04_confirmed_fp.png)

### Mixed and messy — real-world case

**`05_complex_mixed`** — 3 real detected + 1 FP + 1 GT missed entirely. Two `uncertain`, one `reject` (killed a TP), one `confirm` on the FP. The model's every verdict here is wrong in the "helpful" direction.
![Complex mixed](reports/samples/05_complex_mixed.png)

### Missed detections upstream — YOLO's failure, not MedGemma's

**`07_missed_lesion`** — 7 GT lesions, YOLO found only 2 (both confirmed correctly). MedGemma can't fix what the detector never proposed.
![Missed lesions](reports/samples/07_missed_lesion.png)

### Correct nothing — a healthy negative

**`06_true_negative`** — no ground-truth lesions, YOLO returned nothing, no verdict needed.
![True negative](reports/samples/06_true_negative.png)

---

## Full test set at a glance

Every one of the 332 test-set panels, tiled into a single image. Click to view full-resolution (8084 × 6872, 44 MB) — each panel is the same layout as the close-ups above.

[![Collage of all 332 test panels](reports/collage.png)](reports/collage.png)

Individual full-resolution panels + per-image JSONs are in `test_caries_full/testset/`.

---

## Repository layout

```
.
├── analyze.py                  Clinician-facing single-image analysis (image + boxes + findings panel)
├── two_stage.py                Minimal two-stage runner (image → boxes + per-crop verdict)
├── main.py                     Original whole-image localization probe with grounding checks
├── test_full.py                Full test-set evaluation: mAP + veto cross-tab + panels
├── eval_two_stage.py           Two-stage eval used before test_full
├── train_yolo.py               YOLO11 training script
├── prepare_data.py             Build the patient-safe YOLO split from the Zenodo dataset
├── build_vlm_data.py           Convert YOLO split into MedGemma JSONL (with grounded reasoning traces)
├── finetune_medgemma.py        LoRA fine-tune of MedGemma 1.5 4B (CUDA / transformers)
├── colab_medgemma.ipynb        Same fine-tune wrapped for Colab A100
├── merge_lora.py               Merge LoRA into base and save as HF format
├── fix_prefix.py               One-shot: strip transformers-5.x 'model.' prefix from merged safetensors
├── try_caries_medgemma.py      Whole-image inference with the training prompt (against Ollama)
├── make_collage.py             Tile all 332 eval panels into one PNG
├── Modelfile.caries            Ollama Modelfile for the fine-tuned model
├── best.pt / best_m4.pt / best_colab.pt   YOLO11m weights (various runs; best.pt is canonical)
├── yolo11s.pt                  YOLO base weights
├── Dataset/model/
│   └── medgemma_caries_lora/   LoRA adapter (PEFT format)
│   (GGUF weights on HF: sdvzdfgfngdfgh/caries-medgemma-gguf)
├── yolo_split/                 Patient-safe YOLO split (train / val / test)
├── test/                       Pre-finetune baseline eval (base medgemma1.5)
├── test_caries_30/             30-image subset eval (fine-tuned)
├── test_caries_full/           Full 332-image eval (fine-tuned)  ← headline numbers
└── reports/
    ├── collage.png             Full 332-panel collage
    ├── samples/                Close-up representative panels
    └── charts/                 Metrics figures
```

---

## Setup

### Prerequisites
- macOS (tested on Apple Silicon M4) or Linux
- Python 3.11+ with `torch>=2.11`, `transformers>=5.5`, `peft>=0.19`, `ultralytics`, `pillow`, `matplotlib`
- [Ollama](https://ollama.com/) 0.32+ (for serving the quantized fine-tune)
- `git-lfs` for cloning this repo with weights

### Clone

```bash
git lfs install
git clone https://github.com/Eeman1113/dental_study.git
cd dental_study
```

### Get the fine-tuned model into Ollama

GGUF weights live on Hugging Face (GitHub LFS caps files at 2GB; F16 is 7.2GB):
👉 **[sdvzdfgfngdfgh/caries-medgemma-gguf](https://huggingface.co/sdvzdfgfngdfgh/caries-medgemma-gguf)**

```bash
mkdir -p Dataset/model/gguf
huggingface-cli download sdvzdfgfngdfgh/caries-medgemma-gguf \
  medgemma_caries-Q4_K_M.gguf mmproj-medgemma_caries-F16.gguf \
  --local-dir Dataset/model/gguf

ollama create caries-medgemma -f Modelfile.caries
ollama show caries-medgemma
```

Or rebuild from scratch:

```bash
# 1. Merge LoRA into base (needs HF login + license accepted on gated repo)
huggingface-cli login
python merge_lora.py

# 2. Fix transformers-5.x prefix
python fix_prefix.py

# 3. Convert to GGUF (needs a local llama.cpp clone)
git clone https://github.com/ggerganov/llama.cpp .llama.cpp
python .llama.cpp/convert_hf_to_gguf.py Dataset/model/medgemma_caries_merged \
  --outtype f16 --outfile Dataset/model/gguf/medgemma_caries-F16.gguf
python .llama.cpp/convert_hf_to_gguf.py Dataset/model/medgemma_caries_merged \
  --mmproj --outtype f16
mv Dataset/model/medgemma_caries_merged/mmproj-*.gguf \
   Dataset/model/gguf/mmproj-medgemma_caries-F16.gguf

# 4. Quantize + import into Ollama
llama-quantize Dataset/model/gguf/medgemma_caries-F16.gguf \
               Dataset/model/gguf/medgemma_caries-Q4_K_M.gguf Q4_K_M
ollama create caries-medgemma -f Modelfile.caries
```

---

## Run inference

```bash
# Single image, full clinician-facing panel (recommended)
python analyze.py path/to/photo.jpg --model caries-medgemma --out out.png

# Compact two-stage runner (prints verdicts to stdout)
python two_stage.py path/to/photo.jpg --model caries-medgemma

# Whole-image detection with the training prompt
# (mostly collapses to negative — see "what the fine-tune actually does")
python try_caries_medgemma.py path/to/photo.jpg
```

---

## Reproduce the evaluation

```bash
# Full test set (~90 min on M4 Mac mini)
python test_full.py --model caries-medgemma --out ./test_caries_full

# Rebuild the collage of all 332 panels
python make_collage.py

# Regenerate the README charts (needs matplotlib)
python /tmp/make_charts.py  # or copy the script from git history
```

Outputs land in `<out>/METRICS.json`, `<out>/REPORT.md`, and `<out>/testset/*.{png,json}`.

---

## Training

The fine-tune ran on Colab A100-40GB with QLoRA:

| Hyperparameter | Value |
|---|---|
| Base model | `google/medgemma-1.5-4b-it` |
| LoRA rank | 16 |
| LoRA α | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (language tower only; vision tower frozen) |
| Epochs | 3 |
| Batch × grad-accum | 1 × 8 |
| Learning rate | 1e-4 (cosine, warmup 3 %) |
| Precision | bf16 |
| Quantization | 4-bit NF4 |

**Known failure mode:** the training data (built by `build_vlm_data.py`) uses a fixed templated negative response for every image without lesions:
`"Scanning the visible dentition surface by surface. The enamel appears intact with no discoloration, cavitation, or shadowing that would indicate decay."`
If the negative-to-positive ratio in your split is too high, the model memorizes that template and outputs it on every whole-image query. Rebalance the JSONL or paraphrase the negative template before retraining.

The full training scripts are `finetune_medgemma.py` (local CUDA) and `colab_medgemma.ipynb` (Colab A100).

---

## Data source

Original dataset: [Zenodo record 14827784](https://zenodo.org/records/14827784) (~1.6 GB, ~2000 intraoral photographs with pixel-level caries annotations).

`prepare_data.py` builds a patient-safe YOLO split with only Mandibular and Maxillary_Occlusal views (the two views with consistent framing).

---

## License

Code: MIT. Model weights inherit their upstream licenses:
- YOLO11 weights: AGPL-3.0 (Ultralytics)
- MedGemma weights: [HAI-DEF Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms) (Google) — **not for clinical use.**
