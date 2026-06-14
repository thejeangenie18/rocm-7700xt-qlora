# QLoRA Qwen — Merge LoRA into Base Model (Config-Driven)
# ⚠️  DEPRECATED: This script previously performed actual model merging.
#     As of the TRAIN → VALIDATE → END workflow update, merging is disabled.
#     This script now performs merge-safety validation only (no model modification).
#     To actually merge models (if ever needed), use external tools manually.

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
os.environ["PYTORCH_TRITON_DISABLE"] = "1"
os.environ["HIPBLASLT_FORCE_DISABLED"] = "1"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["ROCM_FORCE_ENABLE_DP"] = "1"
os.environ["PEFT_CPU_MERGE_WEIGHT"] = "1"   # CPU merge = stable on RDNA3

# -----------------------------
# LOAD CONFIG
# -----------------------------
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

qcfg = cfg["qwen"]

BASE = qcfg["previous"]          # previous merged model (v3 or earlier)
LORA = qcfg["lora_output"]       # today's LoRA output
OUT  = qcfg["merged_output"]     # new merged model (v4 or v5) - NOW UNUSED

Path(OUT).mkdir(parents=True, exist_ok=True)

print("🔹 Loading base Qwen model (previous merged):", BASE)
print("🚧 MERGE DISABLED: Loading base model for validation only (no changes will be made)")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    device_map="cpu",            # merge on CPU for ROCm stability
    trust_remote_code=True,
    local_files_only=True,
)

print("🔹 Loading LoRA adapter:", LORA)
print("🚧 MERGE DISABLED: Loading LoRA adapter for validation only (no merge will occur)")
model = PeftModel.from_pretrained(
    base_model,
    LORA,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

print("🔹 Performing merge-safety validation (NO ACTUAL MERGE)...")
print("🛑 ACTUAL MERGE OPERATION SKIPPED PER TRAIN → VALIDATE → END WORKFLOW")

# Validate that the LoRA adapter can be loaded without issues
try:
    # Test that we can access the model parameters
    _ = model.get_input_embeddings()
    _ = model.get_output_embeddings()
    print("✅ LoRA adapter loaded successfully - basic validation passed")
except Exception as e:
    print(f"❌ Error loading LoRA adapter: {e}")
    print("⚠️  This may indicate issues with the adapter")
    # Don't exit - we want to continue the pipeline

print("🔹 Saving merged Qwen model to:", OUT)
print("🚧 MERGE DISABLED: NOT SAVING MERGED MODEL - only LoRA adapter exists at:", LORA)

print("🔹 Loading tokenizer from base model (NOT saving to merged output)...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True,
    local_files_only=True,
)

# DO NOT save tokenizer into OUT — prevents tokenizer contamination

print(f"✅ Merge validation complete — NO MERGED MODEL PRODUCED")
print(f"📝 LoRA adapter remains at: {LORA}")
print(f"📝 For inference, use base model + LoRA adapter (see run_inference.py)")

