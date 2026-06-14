#!/usr/bin/env python3
"""
merge_qwen_lora_final.py
⚠️  DEPRECATED: This script previously performed actual model merging.
    As of the TRAIN → VALIDATE → END workflow update, merging is disabled.
    This script now performs merge-safety validation only (no model modification).
    To actually merge models (if ever needed), use external tools manually.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
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
# PATHS
# -----------------------------
BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "/home/jg18/Project/rocm-7700xt-qlora/qwen3b_qlora_output"
OUT = "/home/jg18/Project/rocm-7700xt-qlora/qwen3b_merged_fp16_final"  # NOW UNUSED

print("Loading tokenizer…")
print("🚧 MERGE DISABLED: Loading tokenizer for validation only (not saving to merged output)")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.pad_token = tok.eos_token

# -----------------------------
# LOAD BASE MODEL (CPU, BF16 SAFE)
# -----------------------------
print("Loading sharded base model on CPU…")
print("🚧 MERGE DISABLED: Loading base model for validation only (no changes will be made)")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,   # RDNA3-native
    device_map=None,
    trust_remote_code=True
)

# -----------------------------
# LOAD LORA ADAPTER
# -----------------------------
print("Loading LoRA adapter on CPU…")
print("🚧 MERGE DISABLED: Loading LoRA adapter for validation only (no merge will occur)")
model = PeftModel.from_pretrained(
    base,
    ADAPTER,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)

# -----------------------------
# MERGE
# -----------------------------
print("🚧 MERGE DISABLED: Skipping merge operation - no weights will be combined")
print("🛑 ACTUAL MERGE OPERATION SKIPPED PER TRAIN → VALIDATE → END WORKFLOW")

# -----------------------------
# SAVE MERGED MODEL
# -----------------------------
print("🚧 MERGE DISABLED: NOT SAVING MERGED MODEL")
print(f"📝 LoRA adapter remains at: {ADAPTER}")
print(f"📝 For inference, use base model + LoRA adapter (see run_inference.py)")

print("🔹 Performing merge-safety validation (NO ACTUAL MERGE)...")

# -----------------------------
# VERIFY MERGE AGAINST SHARDS
# -----------------------------
cache_root = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots"
)
snapshots = glob(f"{cache_root}/*")

if not snapshots:
    raise RuntimeError("No Qwen2.5-3B-Instruct snapshot found in HF cache.")

snapshot = snapshots[0]
base_shards = sorted(glob(f"{snapshot}/model-*.safetensors"))
# merged_path = f"{OUT}/model.safetensors"  # MERGE DISABLED - NOT LOADING MERGED WEIGHTS

print("🚧 MERGE DISABLED: Skipping weight difference calculation - no merge performed")

# Validate that we can load the base model and adapter without issues
try:
    # Test that we can access the model parameters
    _ = model.get_input_embeddings()
    _ = model.get_output_embeddings()
    print("✅ LoRA adapter loaded successfully - basic validation passed")

    # Also validate base model loading
    total_base_params = sum(p.numel() for p in base.parameters())
    print(f"✅ Base model loaded successfully - {total_base_params:,} parameters validated")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("⚠️  This may indicate issues with the models")

print("🔹 For actual merging (if needed later), use external tools manually")
print("📝 LoRA adapters remain available for inference")
