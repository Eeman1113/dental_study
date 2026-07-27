# Daath — YOLO + MedGemma caries pipeline

A two-stage detector for dental caries on intraoral photographs:

1. **Stage 1 — YOLO11m** locates candidate lesions (bounding boxes + confidence).
2. **Stage 2 — MedGemma 1.5 4B** (Google, gated) receives each cropped detection and produces a short verdict: `confirm`, `uncertain`, or `reject`, with a one-line rationale.

This repo also ships a **LoRA fine-tune of MedGemma** on the same dataset (labels re-cast as VLM instruction data with grounded reasoning traces), the merged/quantized GGUF weights ready for Ollama, and a full 332-image test-set evaluation comparing YOLO alone, YOLO + base MedGemma, and YOLO + fine-tuned MedGemma.

> ⚠️ **Medical data.** The `yolo_split/`, `test/testset/`, and `test_caries_full/testset/` directories contain anonymized intraoral photographs. Anonymized ≠ consented. If you fork or redistribute, verify the source dataset's terms.

---

## Headline result on the held-out test set (332 images, 850 GT lesions)

Detector: `best.pt` (YOLO11m fine-tuned on a **patient-safe, occlusal-only** split — no patient-level leakage). Verdict model: `medgemma1.5` (base) vs `caries-medgemma` (this fine-tune) via Ollama. Match: IoU ≥ 0.5.

| Pipeline | TP | FP | FN | **Precision** | **Recall** |
|---|---:|---:|---:|---:|---:|
| YOLO alone | 763 | 151 | 87 | **0.835** | **0.898** |
| YOLO + base MedGemma veto | 568 | 79 | 282 | **0.878** | **0.668** |
| YOLO + **fine-tuned** MedGemma veto | 599 | 108 | 251 | **0.847** | **0.705** |

**Detector-only mAP:** mAP50 = 0.860, mAP50-95 = 0.692. Paper's baseline (YOLOv8s on a leaky image-level split): mAP50 = 0.841. Our split is harder because there's no patient leakage.

### What this actually says

- **YOLO alone is the best operational choice.** Both veto configurations trade far more recall than they buy in precision.
  - Base MedGemma: +4.3 pp precision for **−22.9 pp recall**.
  - Fine-tuned MedGemma: +1.2 pp precision for **−19.3 pp recall**.
- **The fine-tune is measurably distinct from the base**, but not usefully better as a veto:
  - It's *more permissive* (rejects only 28.5 % of FPs vs 47.7 %) — keeps more TPs alive, but also lets more FPs through.
  - Verdict counts (fine-tuned): 560 confirm (16 % wrong) · 147 uncertain (14 % wrong) · 207 reject (**79 % of rejects were actually TPs**).
- **Whole-image detection collapsed.** When asked to output boxes on a full image (matching the training prompt), the fine-tune deterministically emits the negative-example template across every image × every seed × every temperature. Almost certainly a training-data imbalance (too many negatives). See `test_full.log` / `test_caries_full/REPORT.md` for the numbers.
- For a **screening** tool you want everything YOLO finds surfaced with reasoning attached, not filtered by a bad veto. `analyze.py` produces that panel.

---

## Full test-set panel collage (332 images)

Every panel below = one test image → original + YOLO detections + MedGemma verdict per detection.

![Collage of all 332 test panels](reports/collage.png)

Full-resolution individual panels are in `test_caries_full/testset/`.

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
├── best.pt / best_m4.pt / best_colab.pt   YOLO11m weights (various training runs; best.pt is canonical)
├── yolo11s.pt                  YOLO base weights
├── Dataset/                    Dataset + trained models
│   └── model/
│       └── medgemma_caries_lora/       LoRA adapter (PEFT format)
│       (GGUF weights on HF: sdvzdfgfngdfgh/caries-medgemma-gguf)
├── yolo_split/                 Patient-safe YOLO split (train / val / test)
├── test/                       Pre-finetune baseline eval (base medgemma1.5)
├── test_caries_30/             30-image subset eval (fine-tuned)
├── test_caries_full/           Full 332-image eval (fine-tuned)  ← headline numbers
└── reports/collage.png         The 332-image panel collage embedded above
```

---

## Setup

### Prerequisites
- macOS (tested on Apple Silicon) or Linux
- Python 3.11+ with `torch`, `transformers>=5.5`, `peft>=0.19`, `ultralytics`, `pillow`
- [Ollama](https://ollama.com/) 0.32+ (for serving the quantized fine-tune)
- Optional: `git-lfs` for cloning this repo with weights

### Clone

```bash
git lfs install
git clone https://github.com/Eeman1113/dental_study.git
cd dental_study
```

### Get the fine-tuned model into Ollama

GGUF weights live on Hugging Face (GitHub LFS caps files at 2GB, F16 is 7.2GB):
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
# 1. Merge LoRA into base (needs HF login + license accepted on the gated repo)
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
# Single image, clinician-facing panel
python analyze.py path/to/photo.jpg --model caries-medgemma --out out.png

# Compact two-stage runner
python two_stage.py path/to/photo.jpg --model caries-medgemma

# Whole-image detection with the training prompt (mostly collapses to negative — see caveats)
python try_caries_medgemma.py path/to/photo.jpg
```

## Reproduce the evaluation

```bash
# Full test set (~90 min on M4)
python test_full.py --model caries-medgemma --out ./test_caries_full

# Then rebuild the collage
python make_collage.py
```

Outputs land in `<out>/METRICS.json` + `<out>/REPORT.md` + `<out>/testset/*.png`.

---

## Training

The fine-tune ran on Colab A100-40GB with QLoRA (r=16, α=32, dropout=0.05) targeting `q,k,v,o,gate,up,down` projections on the language tower only (vision tower frozen). 3 epochs, batch 1 × grad-accum 8, cosine schedule, warmup 3 %. See `finetune_medgemma.py` / `colab_medgemma.ipynb`.

**Known failure mode:** the training data (built by `build_vlm_data.py`) uses a fixed templated negative response (`"Scanning the visible dentition surface by surface. The enamel appears intact..."`) for every image without lesions. If the negative-to-positive ratio in your split is too high, the model memorizes that template and outputs it on every image. That's exactly what happened here — see the headline caveat. Rebalance the JSONL or paraphrase the negative template before retraining.

---

## Data source

Original dataset: [Zenodo record 14827784](https://zenodo.org/records/14827784) (~1.6 GB). See `colab_medgemma.ipynb` cell 4 for the download URL.

## License

Code: MIT. Model weights inherit their upstream licenses:
- YOLO11 weights: AGPL-3.0 (Ultralytics)
- MedGemma weights: [HAI-DEF Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms) (Google) — **not for clinical use.**
