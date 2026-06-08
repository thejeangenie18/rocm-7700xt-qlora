#!/usr/bin/env python3
"""
validate_adapter.py
RDNA3‑safe validation script for merged bf16/fp16 QLoRA models.
Runs a set of prompts and logs outputs to JSONL.
"""

import os
import json
import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# RDNA3 / ROCm STABILITY SETTINGS
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HSA_ENABLE_SDMA"] = "1"
os.environ["ROCM_FORCE_ENABLE_DP"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "./qwen3b_merged_fp16_final"
LOG_DIR = Path("./validation_logs")
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = LOG_DIR / f"validation_{timestamp}.jsonl"

# -----------------------------
# Prompts
# -----------------------------
prompts = {
    "sanity": [
        "Explain what QLoRA is in one paragraph.",
        "Summarize the concept of quantization in simple terms."
    ],
    "accessibility": [
        "Explain spoon theory in simple language.",
        "Rewrite this in plain language: 'Executive dysfunction can impact task initiation.'"
    ],
    "edge_cases": [
        "Write a sentence with emojis.",
        "Explain a concept using only short sentences.",
        "Give me a numbered list of three items."
    ]
}

# -----------------------------
# Load merged model (CPU, BF16 compute)
# -----------------------------
print("[INFO] Loading merged model on CPU (bf16 compute)…")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,   # RDNA3-native
    device_map={"": "cpu"},
    trust_remote_code=True,
)

model.eval()

# -----------------------------
# JSONL-style prompt formatting
# -----------------------------
def format_instruction(instruction: str) -> str:
    escaped = instruction.replace('"', '\\"')
    return f'{{"instruction": "{escaped}", "input": "", "output": "'


# -----------------------------
# Generate function
# -----------------------------
def run(prompt):
    text = format_instruction(prompt)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the output field
    if '"output": "' in decoded:
        decoded = decoded.split('"output": "', 1)[-1]
    if decoded.endswith('"'):
        decoded = decoded[:-1]

    return decoded.strip()

# -----------------------------
# Run validation
# -----------------------------
with open(log_file, "w") as f:
    for category, plist in prompts.items():
        for p in plist:
            out = run(p)
            record = {
                "timestamp": timestamp,
                "category": category,
                "prompt": p,
                "output": out
            }
            f.write(json.dumps(record) + "\n")
            print(f"[OK] {category}: {p[:40]}...")

print(f"\n[COMPLETE] Validation logs saved to: {log_file}")
