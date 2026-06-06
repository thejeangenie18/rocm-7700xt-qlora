#!/usr/bin/env python3
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
OUT = "/home/jg18/Project/rocm-7700xt-qlora/qwen3b_merged_fp16_final"

print("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.pad_token = tok.eos_token

print("Loading sharded base model on CPU…")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map=None,
    trust_remote_code=True
)

print("Loading LoRA adapter on CPU…")
model = PeftModel.from_pretrained(
    base,
    ADAPTER,
    torch_dtype=torch.float16,
    local_files_only=True
)

print("Merging LoRA into base model…")
merged = model.merge_and_unload()

print("Saving merged model as a single safetensors file…")
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)

print("Verifying merge…")

# Find base model shards
cache_root = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots"
)
snapshots = glob(f"{cache_root}/*")

if not snapshots:
    raise RuntimeError("No Qwen2.5-3B-Instruct snapshot found in HF cache.")

snapshot = snapshots[0]
base_shards = sorted(glob(f"{snapshot}/model-*.safetensors"))
merged_path = f"{OUT}/model.safetensors"

# Load merged model
merged_weights = load_file(merged_path)

# Compare against each shard
diffs = 0
total = 0

for shard in base_shards:
    base_weights = load_file(shard)
    for k in base_weights:
        if k in merged_weights:
            b = base_weights[k]
            m = merged_weights[k]
            total += b.numel()
            diffs += (b != m).sum().item()

print("Total elements:", total)
print("Different elements:", diffs)
print("Percent changed:", diffs / total)
print(f"Merged model saved to: {OUT}")
