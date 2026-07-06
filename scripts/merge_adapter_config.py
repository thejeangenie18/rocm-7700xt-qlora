#!/usr/bin/env python3
"""
scripts/merge_adapter_config.py
Config-file-driven merge-safety validation for any HuggingFace model and LoRA adapter.
Merging is disabled per the TRAIN to VALIDATE to END workflow; validation only.
To actually merge models, use external tools manually.

Expected config.yaml structure:
  model:
    previous: /path/to/base-or-previous-merged-model
    lora_output: /path/to/lora-adapter
    merged_output: /path/to/intended-output (unused; no merge performed)
"""

import os
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel

# -----------------------------
# GLOBAL SEED FOR REPRODUCIBILITY
# -----------------------------
set_seed(42)

# -----------------------------
# RDNA3 SAFE ENVIRONMENT
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HSA_ENABLE_SDMA"] = "1"
os.environ["ROCM_FORCE_ENABLE_DP"] = "1"
os.environ["PYTORCH_TRITON_DISABLE"] = "1"
os.environ["HIPBLASLT_FORCE_DISABLED"] = "1"   # force hipBLAS fallback for merge stability
os.environ["PEFT_CPU_MERGE_WEIGHT"] = "1"       # CPU merge = stable on RDNA3

# -----------------------------
# LOAD CONFIG
# -----------------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Config key is configurable; defaults to "model" for general use.
# Override with MODEL_CONFIG_KEY env var if your config uses a different key.
config_key = os.getenv("MODEL_CONFIG_KEY", "model")
if config_key not in cfg:
    raise KeyError(
        f"Config key '{config_key}' not found in config.yaml. "
        f"Set MODEL_CONFIG_KEY to match your config structure. "
        f"Available keys: {list(cfg.keys())}"
    )

mcfg = cfg[config_key]

BASE = mcfg["previous"]          # base or previously merged model
LORA = mcfg["lora_output"]       # LoRA adapter to validate
OUT  = mcfg["merged_output"]     # intended output path (unused)

Path(OUT).mkdir(parents=True, exist_ok=True)

print(f"Loading base model for validation: {BASE}")
print("[MERGE DISABLED] No changes will be made to the model.")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    device_map="cpu",            # merge on CPU for ROCm stability
    trust_remote_code=True,
    local_files_only=True,
)

print(f"Loading LoRA adapter for validation: {LORA}")
print("[MERGE DISABLED] Adapter will be loaded but not merged.")
model = PeftModel.from_pretrained(
    base_model,
    LORA,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

print("[MERGE DISABLED] Skipping merge - no weights will be combined.")
print("[MERGE DISABLED] ACTUAL MERGE SKIPPED PER TRAIN TO VALIDATE TO END WORKFLOW")

# Validate adapter load
try:
    _ = model.get_input_embeddings()
    _ = model.get_output_embeddings()
    print("[SUCCESS] LoRA adapter loaded successfully - basic validation passed")
except Exception as e:
    print(f"[ERROR] Error loading LoRA adapter: {e}")
    print("[WARNING] This may indicate issues with the adapter")

print(f"[MERGE DISABLED] NOT SAVING MERGED MODEL - only LoRA adapter exists at: {LORA}")

print("[INFO] Loading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True,
    local_files_only=True,
)

# DO NOT save tokenizer into OUT - prevents tokenizer contamination

print("[DONE] Merge validation complete. NO MERGED MODEL PRODUCED.")
print(f"LoRA adapter remains at: {LORA}")
print(f"For inference, use base model + LoRA adapter (see scripts/run_inference.py)")
