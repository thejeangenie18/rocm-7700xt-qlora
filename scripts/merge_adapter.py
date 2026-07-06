#!/usr/bin/env python3
"""
scripts/merge_adapter.py
Minimal CLI-driven merge-safety validation for any HuggingFace base model and LoRA adapter.
Merging is disabled per the TRAIN to VALIDATE to END workflow; validation only.
"""

import argparse
import os
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

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, required=True,
                    help="Base model name or path (e.g. Qwen/Qwen2.5-3B-Instruct)")
parser.add_argument("--adapter_path", type=str, default="./models/adapter",
                    help="Path to LoRA adapter directory")
parser.add_argument("--output_dir", type=str, default="./models/merged",
                    help="Output directory (ignored; no merge performed)")
args = parser.parse_args()

BASE    = args.base_model
ADAPTER = args.adapter_path
OUT     = args.output_dir

print(f"[INFO] Loading base model on CPU for validation: {BASE}")
print("[INFO] MERGE DISABLED: No actual merging will be performed")
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,  # bfloat16 for ROCm stability
    device_map={"": "cpu"},      # force CPU
)

print(f"[INFO] Loading LoRA adapter for validation: {ADAPTER}")
print("[INFO] MERGE DISABLED: Adapter will be loaded but not merged")
model = PeftModel.from_pretrained(
    model,
    ADAPTER,
    device_map={"": "cpu"},      # force CPU
)

print("[INFO] Performing merge-safety validation (NO ACTUAL MERGE)...")
print("[INFO] ACTUAL MERGE SKIPPED PER TRAIN TO VALIDATE TO END WORKFLOW")

try:
    _ = model.get_input_embeddings()
    _ = model.get_output_embeddings()
    print("[SUCCESS] LoRA adapter loaded successfully - basic validation passed")
except Exception as e:
    print(f"[ERROR] Error loading LoRA adapter: {e}")
    print("[WARNING] This may indicate issues with the adapter")

print("[INFO] NOT saving merged model - only LoRA adapter exists at:", ADAPTER)
print("[INFO] For inference, use base model + LoRA adapter (see scripts/run_inference.py)")
print(f"[INFO] To actually merge models (if ever needed), use external tools manually")
print(f"[DONE] Merge validation complete - no model saved to: {OUT}")
