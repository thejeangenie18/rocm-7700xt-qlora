import os
import torch
import quanto
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE = "data/train.jsonl"
OUTPUT_DIR = "models/qwen25_3b_lora"

BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4
MAX_STEPS = 200
MAX_LENGTH = 384

# ROCm-safe dtype/optimizer fallback
SUPPORTS_BF16 = getattr(torch, "is_bf16_supported", lambda: False)()
TORCH_DTYPE = torch.bfloat16 if SUPPORTS_BF16 else torch.float16
USE_BF16 = SUPPORTS_BF16
USE_FP16 = not SUPPORTS_BF16
OPTIMIZER = "adamw_torch"

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------------
# Dataset
# -----------------------------
raw = load_dataset("json", data_files=DATA_FILE, split="train")

def format_example(example):
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")

    prompt = instr
    if inp:
        prompt += f"\n\n### Input:\n{inp}"
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{out}"
    return {"text": text}

train = raw.map(format_example)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

train_tok = train.map(tokenize, batched=True, remove_columns=["text"])
train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

# -----------------------------
# Model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    device_map="auto",
)

# -----------------------------
# Quantize linear layers only
# -----------------------------
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        quanto.quantize_module(module, weights=quanto.qint4)

# Freeze base model; only LoRA trains
for param in model.parameters():
    param.requires_grad = False

# -----------------------------
# LoRA config
# -----------------------------
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "w1", "w2", "w3",
    ],
    use_rslora=True,
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# -----------------------------
# Trainer
# -----------------------------
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    max_steps=MAX_STEPS,
    fp16=USE_FP16,
    bf16=USE_BF16,
    gradient_checkpointing=True,
    optim=OPTIMIZER,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    data_collator=collator,
)

trainer.train()

model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
tokenizer.save_pretrained(OUTPUT_DIR)
