# RDNA3 Stability Notes
# This script includes ROCm environment overrides and RDNA3 fixes inspired by
# the BEATEK_ROCm project by Beat‑k:
# https://github.com/Beat-k/BEATEK_ROCm
# These settings are confirmed stable across TinyLlama, Qwen3B incremental,
# and Spoonie‑Helper v5 training runs.

import os
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# -----------------------------
# RDNA3 / ROCm STABILITY SETTINGS
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HSA_ENABLE_SDMA"] = "1"
os.environ["ROCM_FORCE_ENABLE_DP"] = "1"

# -----------------------------
# ABSOLUTE PATHS FOR CLEAN PROJECT STRUCTURE
# -----------------------------
BASE_DIR = "/home/jg18/Project/Qlora"

MODEL_NAME = f"{BASE_DIR}/models/qwen-base"
DATA_PATH = f"{BASE_DIR}/data/ada.jsonl"
OUTPUT_DIR = f"{BASE_DIR}/loras/spoonie-helper-lora"
MAX_SEQ_LEN = 2048

# -----------------------------
# PRE‑RUN SNAPSHOTS
# -----------------------------
os.system(f"rocm-smi > {BASE_DIR}/training/rocm_smi_pre.txt")
os.system(f"cat /proc/meminfo > {BASE_DIR}/training/system_stats_pre.txt")

# -----------------------------
# LOAD TOKENIZER + MODEL (NO TRITON, NO FP16)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # RDNA3‑native, stable
    device_map="auto",           # modern HIP placement
)

model.config.pad_token_id = tokenizer.eos_token_id

# -----------------------------
# LOAD DATASET
# -----------------------------
hf_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print("Loaded", len(hf_dataset), "training examples.")

# -----------------------------
# FORMAT EXAMPLES
# -----------------------------
def format_example(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    if input_text:
        prompt = (
            f"### Instruction\n{instruction}\n"
            f"### Input\n{input_text}\n"
            f"### Output\n"
        )
    else:
        prompt = (
            f"### Instruction\n{instruction}\n"
            f"### Output\n"
        )

    return {"text": prompt + output_text}

hf_dataset = hf_dataset.map(format_example)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )

hf_dataset = hf_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=hf_dataset.column_names,
)

# -----------------------------
# DATA COLLATOR
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# -----------------------------
# QLoRA CONFIG
# -----------------------------
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

# -----------------------------
# TRAINING ARGS (RDNA3‑SAFE)
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,   # RDNA3 register pressure fix
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,                       # RDNA3‑native
    fp16=False,                      # unsafe on RDNA3
    logging_steps=10,
    save_steps=500,
    max_steps=-1,
    save_total_limit=3,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
)

# -----------------------------
# TRAIN
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    data_collator=data_collator,
)

start = time.time()
metrics = trainer.train()
end = time.time()

print("Training metrics:", metrics)
print("Runtime (sec):", end - start)

# -----------------------------
# SAVE
# -----------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# -----------------------------
# POST‑RUN SNAPSHOTS
# -----------------------------
os.system(f"rocm-smi > {BASE_DIR}/training/rocm_smi_post.txt")
os.system(f"cat /proc/meminfo > {BASE_DIR}/training/system_stats_post.txt")

print("Training complete. LoRA saved to:", OUTPUT_DIR)

