#!/usr/bin/env python3
import os
from glob import glob
from safetensors.torch import load_file

# -----------------------------
# PATHS (update if needed)
# -----------------------------
OUT = "/home/jg18/Project/rocm-7700xt-qlora/qwen3b_merged_fp16_final"

# -----------------------------
# LOCATE BASE MODEL IN HF CACHE
# -----------------------------
cache_root = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots"
)
snapshots = glob(f"{cache_root}/*")

if not snapshots:
    raise RuntimeError(
        "❌ Could not find Qwen2.5-3B-Instruct in HF cache.\n"
        "You must load the base model at least once before running this script."
    )

snapshot = snapshots[0]

# Prefer sharded weights if present
sharded = sorted(glob(f"{snapshot}/model-*.safetensors"))
single_file = f"{snapshot}/model.safetensors"

if sharded:
    print("Found sharded base model.")
    base_paths = sharded
elif os.path.exists(single_file):
    print("Found single-file base model.")
    base_paths = [single_file]
else:
    raise RuntimeError("❌ No model.safetensors or model-*.safetensors found in snapshot.")

print("Using base model files:")
for p in base_paths:
    print("  ", p)

# -----------------------------
# LOAD MERGED MODEL
# -----------------------------
merged_path = f"{OUT}/model.safetensors"

if not os.path.exists(merged_path):
    raise RuntimeError(f"❌ Merged model not found at: {merged_path}")

print("\nLoading merged model:", merged_path)
merged_weights = load_file(merged_path)

# -----------------------------
# STRICT DIFF CHECK
# -----------------------------
diffs = 0
total = 0

print("\nComparing weights…")

for base_path in base_paths:
    base_weights = load_file(base_path)

    for k in base_weights:
        if k in merged_weights:
            b = base_weights[k]
            m = merged_weights[k]

            total += b.numel()
            diffs += (b != m).sum().item()

percent = diffs / total if total > 0 else 0.0

# -----------------------------
# RESULTS
# -----------------------------
print("\n================ MERGE VERIFICATION ================")
print("Total elements:", total)
print("Different elements:", diffs)
print("Percent changed:", percent)
print("Merged model directory:", OUT)
print("====================================================")
