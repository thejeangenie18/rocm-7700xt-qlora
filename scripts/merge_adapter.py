#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--adapter_path", type=str, default="./models/adapter")
parser.add_argument("--output_dir", type=str, default="./models/merged")
args = parser.parse_args()

BASE = args.base_model
ADAPTER = args.adapter_path
OUT = args.output_dir

print(f"[INFO] Loading base model on CPU: {BASE}")
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},   # FORCE CPU
)

print(f"[INFO] Loading LoRA adapter: {ADAPTER}")
model = PeftModel.from_pretrained(
    model,
    ADAPTER,
    device_map={"": "cpu"},   # FORCE CPU
)

print("[INFO] Merging LoRA weights into base model…")
model = model.merge_and_unload()

print("[INFO] Saving merged model…")
model.save_pretrained(OUT)

print("[INFO] Saving tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.save_pretrained(OUT)

print(f"[DONE] Merged model saved to: {OUT}")
