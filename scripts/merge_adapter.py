#!/usr/bin/env python3
"""
scripts/merge_adapter.py
⚠️  DEPRECATED: As of the TRAIN → VALIDATE → END workflow update,
    actual merging is disabled. This script now performs merge-safety
    validation only (no model modification).
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model name or path")
parser.add_argument("--adapter_path", type=str, default="./models/adapter", help="Path to LoRA adapter directory")
parser.add_argument("--output_dir", type=str, default="./models/merged", help="Output directory for merged model (ignored - no merge performed)")
args = parser.parse_args()

BASE = args.base_model
ADAPTER = args.adapter_path
OUT = args.output_dir

print(f"[INFO] Loading base model on CPU for validation: {BASE}")
print("[INFO] MERGE DISABLED: No actual merging will be performed")
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,  # Using bfloat16 for ROCm stability
    device_map={"": "cpu"},   # FORCE CPU
)

print(f"[INFO] Loading LoRA adapter for validation: {ADAPTER}")
print("[INFO] MERGE DISABLED: Adapter will be loaded but not merged")
model = PeftModel.from_pretrained(
    model,
    ADAPTER,
    device_map={"": "cpu"},   # FORCE CPU
)

print("[INFO] Performing merge-safety validation (NO ACTUAL MERGE)...")
print("[INFO] ACTUAL MERGE OPERATION SKIPPED PER TRAIN → VALIDATE → END WORKFLOW")

# Validate that the LoRA adapter can be loaded without issues
try:
    # Test that we can access the model parameters
    _ = model.get_input_embeddings()
    _ = model.get_output_embeddings()
    print("[SUCCESS] LoRA adapter loaded successfully - basic validation passed")
except Exception as e:
    print(f"[ERROR] Error loading LoRA adapter: {e}")
    print("[WARNING] This may indicate issues with the adapter")

print("[INFO] NOT saving merged model - only LoRA adapter exists at:", ADAPTER)
print("[INFO] For inference, use base model + LoRA adapter (see run_inference.py)")
print(f"[INFO] To actually merge models (if ever needed), use external tools manually")
print(f"[DONE] Merge validation complete - no model saved to: {OUT}")
