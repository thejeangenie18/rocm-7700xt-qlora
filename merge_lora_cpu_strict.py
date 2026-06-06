import os
from glob import glob
from safetensors.torch import load_file

# Find the base model safetensors file in the HF cache
cache_root = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots")
snapshots = glob(f"{cache_root}/*")

if not snapshots:
    raise RuntimeError("Could not find Qwen2.5-3B-Instruct in HF cache. Did you load it before?")

base_path = os.path.join(snapshots[0], "model.safetensors")

print("Base model path:", base_path)

base_weights = load_file(base_path)
merged_weights = load_file(f"{OUT}/model.safetensors")

diffs = 0
total = 0

for k in base_weights:
    if k in merged_weights:
        b = base_weights[k]
        m = merged_weights[k]
        total += b.numel()
        diffs += (b != m).sum().item()

print("Total elements:", total)
print("Different elements:", diffs)
print("Percent changed:", diffs / total)

