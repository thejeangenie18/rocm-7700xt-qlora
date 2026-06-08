#!/usr/bin/env python3
"""
smoke_test.py
RDNA3‑safe smoke test: load model on CPU, attach LoRA adapter, run forward/backward.

Usage:
  python tools/smoke_test.py --model Qwen/Qwen2.5-3B-Instruct --adapter ./qwen3b_qlora_output
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=False)
    args = p.parse_args()

    device = "cpu"  # smoke test always runs on CPU

    print("Loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print("Loading base model on CPU (bf16 compute)…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,   # RDNA3-native
        device_map={"": device},
        trust_remote_code=True,
    )

    if args.adapter:
        print("Attaching LoRA adapter…")
        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )

    model.train()  # enable gradients for backward test

    prompt = (
        "### Instruction\nNormalize this vendor entry.\n"
        "### Input\nVendor: Example Co. URL: http://example.com\n"
        "### Output\n"
    )

    print("Running forward pass…")
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model(**inputs)
    print("Forward pass OK. Logits shape:", out.logits.shape)

    if args.adapter:
        print("Running tiny backward pass…")
        loss = out.logits[..., :1].sum()
        loss.backward()
        print("Backward pass OK")

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
