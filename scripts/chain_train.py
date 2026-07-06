"""
chain_train.py
Generalized RDNA3-safe TRAIN to VALIDATE to END pipeline.

Define any number of training and validation stages in the PIPELINE list below,
or pass a JSON config file via --config. The script will run them in order:

    1. Train model A
    2. Validate model A
    3. Train model B
    4. Validate model B
    5. (add more stages as needed)

Works with any HuggingFace model; not tied to a specific architecture.
"""

import argparse
import json
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
#   - "validation_script": path to validation script (optional)
#   - "base_model": base model used for validation (optional; falls back to BASE_MODEL env var)
#
# Example:
#   PIPELINE = [
#       {
#           "name": "My Model Stage 1",
#           "train_script": "scripts/train_rdna3_fix.py",
#           "lora_out": "loras/my-adapter",
#           "validation_script": "tools/validate_all.py",
#       },
#   ]
# ─────────────────────────────────────────────────────────────

PIPELINE = []


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


def run_validation(base_model: str, lora_path: Path, validation_script: str = "tools/validate_all.py") -> bool:
    """Run validation on the trained LoRA adapter."""
    print(f"\n=== Running Validation ===")

    if not os.path.exists(validation_script):
        print(f"[WARNING] Validation script not found: {validation_script}")
        print("[INFO] Skipping validation - assuming success")
        return True

    cmd = f"python {validation_script} --base_model {base_model} --adapter {lora_path}"
    result = run(cmd)

    if result == 0:
        print(f"[SUCCESS] Validation passed for {lora_path}")
        return True
    else:
        print(f"[WARNING] Validation failed for {lora_path}")
        print("[INFO] Continuing pipeline despite validation failure (TRAIN to VALIDATE to END workflow)")
        return True  # Continue regardless; validation failures do not stop the pipeline


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="Optional JSON file defining pipeline stages (overrides the PIPELINE list)")
    args = ap.parse_args()

    pipeline = PIPELINE

    if args.config:
        with open(args.config) as f:
            pipeline = json.load(f)
        print(f"Loaded pipeline config from: {args.config}")

    if not pipeline:
        print("[ERROR] No pipeline stages defined. Populate the PIPELINE list or pass --config.")
        sys.exit(1)

    print("\n──────────────────────────────────────────────")
    print(" RDNA3 Multi-Stage TRAIN to VALIDATE to END Trainer")
    print("──────────────────────────────────────────────\n")

    base_model = os.environ.get("BASE_MODEL", "")

    for stage in pipeline:
        name              = stage["name"]
        train_script      = stage["train_script"]
        lora_out          = Path(stage["lora_out"])
        validation_script = stage.get("validation_script", "tools/validate_all.py")
        stage_base_model  = stage.get("base_model", base_model)

        print(f"\n=== Stage: {name} - Training ===")
        if run(f"python {train_script}") != 0:
            print(f"[ERROR] Training failed for {name}. Aborting pipeline.")
            sys.exit(1)

        print(f"Waiting for LoRA output: {lora_out}")
        if not wait_for_output(lora_out):
            print(f"[ERROR] LoRA output missing for {name}. Aborting.")
            sys.exit(1)

        if stage_base_model:
            run_validation(stage_base_model, lora_out, validation_script)
        else:
            print(f"[INFO] No base_model set for stage '{name}'; skipping validation.")
            print("[INFO] Set BASE_MODEL env var or add 'base_model' to the stage config.")

    print("\n=== All stages complete. ===")
    print("[INFO] Pipeline followed TRAIN to VALIDATE to END workflow.")
    print("[INFO] No merged models were produced - only LoRA adapters are saved.")


if __name__ == "__main__":
    main()
