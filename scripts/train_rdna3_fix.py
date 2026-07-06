# RDNA3 Stability Notes
# This script includes ROCm environment overrides and RDNA3 fixes inspired by
# the BEATEK_ROCm project by Beat-k:
# https://github.com/Beat-k/BEATEK_ROCm

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
# CONFIGURABLE PATHS (override via environment variables)
# -----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
DATA_PATH  = os.getenv("DATA_PATH",  "./datasets/ada.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./loras/adapter")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "2048"))

# Snapshot paths are relative to OUTPUT_DIR
SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# -----------------------------
# PRE-RUN SNAPSHOTS
# -----------------------------
os.system(f"rocm-smi > {SNAPSHOT_DIR}/rocm_smi_pre.txt")
os.system(f"cat /proc/meminfo > {SNAPSHOT_DIR}/system_stats_pre.txt")

# -----------------------------
# LOAD TOKENIZER + MODEL (NO TRITON, NO FP16)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # RDNA3-native, stable
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
# LoRA target_modules below are defaults for Llama/Qwen family models.
# Adjust for other architectures (e.g., Falcon uses query_key_value;
# Phi-2 uses q_proj/v_proj/dense).
# -----------------------------
lora_config = LoraConfig(
    r=int(os.getenv("LORA_RANK", "64")),
    lora_alpha=int(os.getenv("LORA_ALPHA", "16")),
    lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)

# -----------------------------
# TRAINING ARGS (RDNA3-SAFE)
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,   # RDNA3 register pressure fix
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,                       # RDNA3-native
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
# POST-RUN SNAPSHOTS
# -----------------------------
os.system(f"rocm-smi > {SNAPSHOT_DIR}/rocm_smi_post.txt")
os.system(f"cat /proc/meminfo > {SNAPSHOT_DIR}/system_stats_post.txt")

print("Training complete. LoRA saved to:", OUTPUT_DIR)
