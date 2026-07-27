#!/usr/bin/env python3
"""
finetune_medgemma.py - LoRA fine-tune MedGemma 1.5 4B on your caries JSONL.

REALITY CHECK ON HARDWARE:
  Training a 4B multimodal model with a vision tower does NOT fit comfortably in
  16 GB. On your Mac mini this will either OOM or crawl. Two honest options:
    (A) Rent one A100/H100 for a few hours (~cost of lunch) and run THIS script
        with --device cuda. Recommended. Finishes overnight.
    (B) Stay local via Apple's MLX, which handles unified memory better:
            pip install mlx-vlm
            python -m mlx_vlm.lora \
                --model mlx-community/medgemma-1.5-4b-it-bf16 \
                --train data/train.jsonl --batch-size 1 --iters 2000
        (flag names drift between mlx-vlm versions - check `python -m mlx_vlm.lora -h`)

  This script is the CUDA/transformers path (A). It's the robust one to build on.

VERSION SENSITIVITY: the multimodal SFT surface in transformers changes often.
The lines most likely to need adjusting against your installed versions are
tagged  # >>> VERIFY. Check the current MedGemma model card and transformers
docs if any of them throw.

    python finetune_medgemma.py --data ./vlm_data --device cuda

Requires: transformers, peft, accelerate, bitsandbytes (for 4-bit), pillow, torch
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,   # >>> VERIFY: generic loader for Gemma3-family VLMs
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "google/medgemma-1.5-4b-it"   # gated - `huggingface-cli login` first


class CariesVLM(Dataset):
    def __init__(self, jsonl):
        self.rows = [json.loads(l) for l in Path(jsonl).read_text().splitlines() if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collator(processor):
    def collate(batch):
        images, texts = [], []
        for row in batch:
            img = Image.open(row["image"]).convert("RGB")
            # Build the chat with an image placeholder in the user turn.
            msgs = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": row["messages"][0]["content"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": row["messages"][1]["content"]},
                ]},
            ]
            # >>> VERIFY: apply_chat_template signature / add_generation_prompt behaviour
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=False)
            images.append([img])
            texts.append(text.strip())

        batch_enc = processor(
            text=texts, images=images,
            return_tensors="pt", padding=True,
        )
        labels = batch_enc["input_ids"].clone()

        # Mask pad tokens.
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # Mask image-token positions so loss is computed on text only.
        # >>> VERIFY: image_token_id attribute name varies (image_token_index etc.)
        img_tok = getattr(processor.tokenizer, "image_token_id", None)
        if img_tok is not None:
            labels[labels == img_tok] = -100
        batch_enc["labels"] = labels
        return batch_enc
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="vlm_data dir with train/val jsonl")
    ap.add_argument("--out", default="./medgemma_caries_lora")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--four-bit", action="store_true",
                    help="QLoRA - load base in 4-bit to fit smaller GPUs")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    quant = None
    if args.four_bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None,
        attn_implementation="eager",   # >>> VERIFY: Gemma3 recommends eager attn
    )

    # LoRA on the LANGUAGE tower only. Freeze the vision encoder - you don't have
    # remotely enough data to retrain how it sees, and touching it destabilizes
    # everything. This is the single most important choice in the file.
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # >>> VERIFY: exclude vision-tower module names if the above regex-matches them
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = CariesVLM(Path(args.data) / "train.jsonl")
    val_ds = CariesVLM(Path(args.data) / "val.jsonl")
    collate = make_collator(processor)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=200, save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,   # >>> critical: keeps our dict columns intact
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collate,
    )
    trainer.train()

    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(f"\nLoRA adapter -> {args.out}")
    print("Inference: load base MedGemma + this adapter, then reuse your main.py")
    print("prompt/parsing unchanged. FIRST evaluate on test.jsonl and compare")
    print("mAP against the YOLO baseline before believing any of this helped.")


if __name__ == "__main__":
    main()
