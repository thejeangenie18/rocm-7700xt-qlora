#!/usr/bin/env python3
"""
merge_adapter.py
RDNA3‑safe merge: load base model + LoRA adapter on CPU, merge, save fp16 safetensors.

Usage:
  python tools/merge_adapter.py --base Qwen/Qwen2.5-3B-Instruct \
                                --adapter ./qwen3b_qlora_output \
                                --out ./qwen3b_merged_fp16
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -----------------------------
# RDNA3 / ROCm STABILITY SETTINGS
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HSA_ENABLE_SDMA"] = "1"
os.environ["ROCM_FORCE_ENABLE_DP"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base model name or path")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory")
    p.add_argument("--out", required=True, help="Output directory for merged model")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print("Loading base model on CPU (fp16 weights)…")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        trust_remote_code=True,
    )

    print("Loading LoRA adapter on CPU…")
    model = PeftModel.from_pretrained(
        base,
        args.adapter,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    print("Merging LoRA adapter into base model…")
    merged = model.merge_and_unload()

    print("Saving merged model (safetensors)…")
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)

    print(f"Merged model saved to: {args.out}")

if __name__ == "__main__":
    main()
