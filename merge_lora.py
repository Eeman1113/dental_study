#!/usr/bin/env python3
"""
merge_lora.py - merge the caries LoRA into MedGemma 1.5 4B and save as HF format.

Output goes to ./Dataset/model/medgemma_caries_merged/ ready for llama.cpp
conversion. Requires HF login for the gated google/medgemma-1.5-4b-it repo.

    huggingface-cli login   # once
    python merge_lora.py
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "google/medgemma-1.5-4b-it"
DEFAULT_ADAPTER = "./Dataset/model/medgemma_caries_lora"
DEFAULT_OUT = "./Dataset/model/medgemma_caries_merged"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading base: {BASE}  (this pulls ~8GB the first time)")
    base = AutoModelForImageTextToText.from_pretrained(
        BASE, torch_dtype=dtype, device_map="cpu", low_cpu_mem_usage=True,
    )

    print(f"attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("merging LoRA weights into base and unloading adapter")
    model = model.merge_and_unload()

    print(f"saving merged model to {out}")
    model.save_pretrained(out, safe_serialization=True)

    print(f"saving processor + tokenizer to {out}")
    processor = AutoProcessor.from_pretrained(BASE)
    processor.save_pretrained(out)

    print("done. next: convert to GGUF with llama.cpp.")


if __name__ == "__main__":
    main()
