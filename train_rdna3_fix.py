# RDNA3 Stability Notes
# This script includes ROCm environment overrides and RDNA3 fixes inspired by
# the BEATEK_ROCm project by Beat‑k:
# https://github.com/Beat-k/BEATEK_ROCm
# Their documentation was instrumental in stabilizing QLoRA training on RDNA3 GPUs.

import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# -----------------------------
# RDNA3 / ROCm STABILITY SETTINGS
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HSA_ENABLE_SDMA"] = "1"
os.environ["ROCM_FORCE_ENABLE_DP"] = "1"

# Optional: allow TF32 matmul on ROCm 6.1+ (improves throughput)
torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "data/fourth-train.jsonl"
OUTPUT_DIR = "./qwen3b_qlora_output"
MAX_SEQ_LEN = 2048

# -----------------------------
# 4-BIT QUANTIZATION CONFIG
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# -----------------------------
# LOAD TOKENIZER + MODEL (RDNA3-SAFE)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0},  # force everything on GPU 0
)

model.config.pad_token_id = tokenizer.eos_token_id

# -----------------------------
# LOAD DATASET (JSONL)
# -----------------------------
hf_dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

print("Loaded", len(hf_dataset), "training examples.")

# -----------------------------
# FORMAT EXAMPLES (JSONL-STYLE PROMPT)
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

    full_text = prompt + output_text
    return {"text": full_text}

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
# TRAINING ARGS
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=False,
    fp16=True,
    logging_steps=5,
    save_steps=200,
    max_steps=200,
    save_total_limit=3,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
    seed=42,
)

# -----------------------------
# TRAIN
# -----------------------------
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    data_collator=data_collator,
)

trainer.train()

# -----------------------------
# SAVE
# -----------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Model saved to:", OUTPUT_DIR)