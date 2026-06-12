# QLoRA Qwen — Merge LoRA into Base Model (Config-Driven)

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
OUT  = qcfg["merged_output"]     # new merged model (v4 or v5)

Path(OUT).mkdir(parents=True, exist_ok=True)

print("🔹 Loading base Qwen model (previous merged):", BASE)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    device_map="cpu",            # merge on CPU for ROCm stability
    trust_remote_code=True,
    local_files_only=True,
)

print("🔹 Loading LoRA adapter:", LORA)
model = PeftModel.from_pretrained(
    base_model,
    LORA,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

print("🔹 Merging LoRA → base model (may take 1–5 minutes)...")
merged = model.merge_and_unload()

print("🔹 Saving merged Qwen model to:", OUT)
merged.save_pretrained(OUT, safe_serialization=True)

print("🔹 Loading tokenizer from base model (NOT saving to merged output)...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    trust_remote_code=True,
    local_files_only=True,
)

# DO NOT save tokenizer into OUT — prevents tokenizer contamination

print(f"✅ Merge complete — merged Qwen model saved to {OUT}")

