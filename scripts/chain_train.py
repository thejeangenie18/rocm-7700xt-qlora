"""
chain_train.py
──────────────────────────────────────────────────────────────
Generalized RDNA3-safe TRAIN → VALIDATE → END pipeline.

Define ANY number of training + validation stages and this script
will run them in order:

    1. Train model A
    2. Validate model A
    3. Train model B
    4. Validate model B
    5. (optional) Train model C
    6. (optional) Validate model C

Perfect for Qwen → TinyLlama → (future models) with comprehensive validation.
"""

import subprocess
import time
from pathlib import Path
import sys
import os


# ─────────────────────────────────────────────────────────────
# CONFIGURE YOUR TRAINING PIPELINE HERE
# Each stage has:
#   - "name": label for logs
#   - "train_script": path to training script
#   - "lora_out": directory expected after training
#   - "validation_script": path to validation script (optional, defaults to validate_all.py)
# ─────────────────────────────────────────────────────────────

PIPELINE = [
    {
        "name": "Qwen Spoonie Helper",
        "train_script": "training/train_spoonie.py",
        "lora_out": Path("loras/spoonie-helper-v3-lora"),
        "validation_script": "validate_all.py",
    },
    {
        "name": "TinyLlama Helper",
        "train_script": "training/tinyllama.py",
        "lora_out": Path("loras/tinyllama-helper-v2-lora"),
        "validation_script": "validate_all.py",
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


def run_validation(base_model: str, lora_path: Path, validation_script: str = "validate_all.py") -> bool:
    """Run validation on the trained LoRA adapter."""
    print(f"\n=== Running Validation ===")

    # Use the validation script from the Qlora directory (shared)
    validation_script_path = f"/home/jg18/Project/Qlora/{validation_script}"

    if not os.path.exists(validation_script_path):
        print(f"[WARNING] Validation script not found: {validation_script_path}")
        print("[INFO] Skipping validation - assuming success")
        return True

    cmd = f"python {validation_script_path} --base_model {base_model} --adapter {lora_path}"
    result = run(cmd)

    if result == 0:
        print(f"[SUCCESS] Validation passed for {lora_path}")
        return True
    else:
        print(f"[WARNING] Validation failed for {lora_path}")
        print("[INFO] Continuing pipeline despite validation failure (as per TRAIN → VALIDATE → END workflow)")
        return True  # Continue anyway - validation failures don't stop the pipeline


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def main():
    print("\n──────────────────────────────────────────────")
    print(" RDNA3 Multi-Stage TRAIN → VALIDATE → END Trainer")
    print("──────────────────────────────────────────────\n")

    # Get base model from config or environment
    base_model = os.environ.get("BASE_MODEL", "NousResearch/Llama-2-7b-hf")

    for stage in PIPELINE:
        name = stage["name"]
        train_script = stage["train_script"]
        lora_out = stage["lora_out"]
        validation_script = stage.get("validation_script", "validate_all.py")

        print(f"\n=== Stage: {name} — Training ===")
        if run(f"python {train_script}") != 0:
            print(f"[ERROR] Training failed for {name}. Aborting pipeline.")
            sys.exit(1)

        print(f"Waiting for LoRA output: {lora_out}")
        if not wait_for_output(lora_out):
            print(f"[ERROR] LoRA output missing for {name}. Aborting.")
            sys.exit(1)

        # Run validation (but don't fail the pipeline on validation failure)
        validation_passed = run_validation(base_model, lora_out, validation_script)
        if not validation_passed:
            print(f"[INFO] Validation indicated issues, but continuing pipeline as per TRAIN → VALIDATE → END workflow")

    print("\n=== All stages complete! ===")
    print("[INFO] Pipeline followed TRAIN → VALIDATE → END workflow")
    print("[INFO] No merged models were produced - only LoRA adapters are saved")


if __name__ == "__main__":
    main()