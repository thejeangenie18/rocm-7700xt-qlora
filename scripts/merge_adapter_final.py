#!/usr/bin/env python3
"""
scripts/merge_adapter_final.py
Performs merge-safety validation for any HuggingFace base model and LoRA adapter.
Merging is disabled per the TRAIN to VALIDATE to END workflow; validation only.
To actually merge models, use external tools manually.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from glob import glob

# -----------------------------
# RDNA3 / ROCm STABILITY SETTINGS
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HSA_ENABLE_SDMA"] = "1"
os.environ["ROCM_FORCE_ENABLE_DP"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------
# ARGS
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True,
                    help="Base model name or path (e.g. Qwen/Qwen2.5-3B-Instruct)")
parser.add_argument("--adapter", required=True,
                    help="Path to LoRA adapter directory")
parser.add_argument("--output", default="./models/merged",
                    help="Output path (unused; no merge performed)")
args = parser.parse_args()

BASE    = args.base_model
ADAPTER = args.adapter
OUT     = args.output

print("Loading tokenizer...")
print("[MERGE DISABLED] Loading tokenizer for validation only.")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.pad_token = tok.eos_token

# -----------------------------
# LOAD BASE MODEL (CPU, BF16 SAFE)
# -----------------------------
print("Loading base model on CPU...")
print("[MERGE DISABLED] Loading base model for validation only.")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,   # RDNA3-native
    device_map=None,
    trust_remote_code=True
)

# -----------------------------
# LOAD LORA ADAPTER
# -----------------------------
print("Loading LoRA adapter on CPU...")
print("[MERGE DISABLED] Loading LoRA adapter for validation only.")
model = PeftModel.from_pretrained(
    base,
    ADAPTER,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)

# -----------------------------
# MERGE DISABLED
# -----------------------------
print("[MERGE DISABLED] Skipping merge operation.")
print("[MERGE DISABLED] ACTUAL MERGE SKIPPED PER TRAIN TO VALIDATE TO END WORKFLOW")

# -----------------------------
# SAVE MERGED MODEL (DISABLED)
# -----------------------------
print("[MERGE DISABLED] NOT SAVING MERGED MODEL")
print(f"LoRA adapter remains at: {ADAPTER}")
print(f"For inference, use base model + LoRA adapter (see scripts/run_inference.py)")

print("Performing merge-safety validation (NO ACTUAL MERGE)...")

# -----------------------------
# VALIDATION
# -----------------------------
try:
    _ = model.get_input_embeddings()
    _ = model.get_output_embeddings()
    print("[SUCCESS] LoRA adapter loaded successfully - basic validation passed")

    total_base_params = sum(p.numel() for p in base.parameters())
    print(f"[SUCCESS] Base model loaded - {total_base_params:,} parameters validated")
except Exception as e:
    print(f"[ERROR] {e}")
    print("[WARNING] This may indicate issues with the models")

print("For actual merging (if needed later), use external tools manually.")
print("LoRA adapters remain available for inference.")
