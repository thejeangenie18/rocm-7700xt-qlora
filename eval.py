#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# LOAD BASE MODEL (BF16, RDNA3-SAFE)
# -----------------------------
BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "releases/tinyllama-qlora/final"

print("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.pad_token = tok.eos_token

print("Loading base model…")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,   # RDNA3-native
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter…")
model = PeftModel.from_pretrained(
    base,
    ADAPTER,
    torch_dtype=torch.bfloat16,
)

# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def infer(prompt):
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
    )

    print(tok.decode(out[0], skip_special_tokens=True))
    print("-" * 80)

# -----------------------------
# TEST SUITE
# -----------------------------
tests = [
    # SQL-only enforcement
    """### Instruction
Respond only with SQL.
### Input
List all spoonie-owned stores.
### Output
""",

    # Structured query
    """### Instruction
Execute structured query.
### Input
{"action":"query","filter":{"vendor":"Target"}}
### Output
""",

    # Natural language → SQL
    """### Instruction
Respond only with SQL.
### Input
Show me all stores that accept Medicaid.
### Output
""",

    # JSONB filter
    """### Instruction
Respond only with SQL.
### Input
Find all stores where specs->>'color' = 'blue'.
### Output
""",

    # Array membership
    """### Instruction
Respond only with SQL.
### Input
List all black-owned vendors.
### Output
""",

    # Error handling
    """### Instruction
Execute structured query.
### Input
{}
### Output
"""
]

for t in tests:
    infer(t)
