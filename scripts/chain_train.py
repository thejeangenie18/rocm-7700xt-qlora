"""
chain_train.py
──────────────────────────────────────────────────────────────
Generalized RDNA3-safe multi-stage training pipeline.

Define ANY number of training + merge stages and this script
will run them in order:

    1. Train model A
    2. Merge model A
    3. Train model B
    4. Merge model B
    5. (optional) Train model C
    6. (optional) Merge model C

Perfect for Qwen → TinyLlama → (future models).
"""

import subprocess
import time
from pathlib import Path
import sys


# ─────────────────────────────────────────────────────────────
# CONFIGURE YOUR TRAINING PIPELINE HERE
# Each stage has:
#   - "name": label for logs
#   - "train_script": path to training script
#   - "merge_script": path to merge script (optional)
#   - "lora_out": directory expected after training
# ─────────────────────────────────────────────────────────────

PIPELINE = [
    {
        "name": "Qwen Spoonie Helper",
        "train_script": "training/train_spoonie.py",
        "merge_script": "training/merge_qwen.py",
        "lora_out": Path("loras/spoonie-helper-v3-lora"),
    },
    {
        "name": "TinyLlama Helper",
        "train_script": "training/tinyllama.py",
        "merge_script": "training/merge_tinyllama.py",
        "lora_out": Path("loras/tinyllama-helper-v2-lora"),
    },
]


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def run(cmd: str) -> int:
    print(f"\n=== Running: {cmd} ===\n")
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    return process.returncode


def wait_for_output(path: Path, timeout=900) -> bool:
    """Wait up to 15 minutes for LoRA output to appear."""
    start = time.time()
    while time.time() - start < timeout:
        if path.exists():
            return True
        time.sleep(5)
    return False


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def main():
    print("\n──────────────────────────────────────────────")
    print(" RDNA3 Multi-Stage Chain Trainer")
    print("──────────────────────────────────────────────\n")

    for stage in PIPELINE:
        name = stage["name"]
        train_script = stage["train_script"]
        merge_script = stage.get("merge_script")
        lora_out = stage["lora_out"]

        print(f"\n=== Stage: {name} — Training ===")
        if run(f"python {train_script}") != 0:
            print(f"[ERROR] Training failed for {name}. Aborting pipeline.")
            sys.exit(1)

        print(f"Waiting for LoRA output: {lora_out}")
        if not wait_for_output(lora_out):
            print(f"[ERROR] LoRA output missing for {name}. Aborting.")
            sys.exit(1)

        if merge_script:
            print(f"\n=== Stage: {name} — Merging ===")
            if run(f"python {merge_script}") != 0:
                print(f"[ERROR] Merge failed for {name}. Aborting pipeline.")
                sys.exit(1)

    print("\n=== All stages complete! ===\n")


if __name__ == "__main__":
    main()

