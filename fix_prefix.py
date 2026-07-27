#!/usr/bin/env python3
"""
fix_prefix.py - strip leading 'model.' from tensor names in the merged model.

transformers 5.x + PEFT.save_pretrained wraps everything with an extra 'model.'
prefix that llama.cpp's converter can't map. This rewrites the safetensors
file in-place with the vanilla HF layout ('language_model.model.*',
'vision_tower.*', 'multi_modal_projector.*').
"""
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("./Dataset/model/medgemma_caries_merged/model.safetensors")
TMP = SRC.with_suffix(".safetensors.tmp")

metadata = {}
tensors = {}
with safe_open(SRC, framework="pt") as f:
    meta = f.metadata()
    if meta:
        metadata.update(meta)
    for k in f.keys():
        new_k = k[len("model."):] if k.startswith("model.") else k
        tensors[new_k] = f.get_tensor(k)

save_file(tensors, TMP, metadata=metadata or None)
TMP.replace(SRC)
print(f"rewrote {SRC} ({len(tensors)} tensors)")
print("sample keys:")
for k in list(tensors.keys())[:3]:
    print(" ", k)
