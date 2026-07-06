#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# RDNA3 / ROCm STABILITY SETTINGS
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HSA_ENABLE_SDMA"] = "1"
os.environ["ROCM_FORCE_ENABLE_DP"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------
# ARGS
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", required=True, help="Base model name or local path")
parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
args = parser.parse_args()

BASE    = args.base_model
ADAPTER = args.adapter

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.pad_token = tok.eos_token

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,   # RDNA3-native
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
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
# Prompts cover general instruction following, accessibility rewriting,
# and structured output. Adjust or extend for your fine-tuning objective.
# -----------------------------
tests = [
    # General instruction following
    """### Instruction
Explain what a neural network is in one short paragraph.
### Input

### Output
""",

    # Accessibility rewriting
    """### Instruction
Rewrite the following in plain language.
### Input
Executive dysfunction can impact task initiation and completion.
### Output
""",

    # Summarization
    """### Instruction
Summarize the following in two sentences.
### Input
Spoon theory is a metaphor used by people with chronic illness or disability to explain their limited daily energy. Each activity costs "spoons", and once they are gone, the person cannot continue without rest.
### Output
""",

    # Structured output
    """### Instruction
List three tips for writing accessible documentation.
### Input

### Output
""",

    # Short-sentence constraint
    """### Instruction
Explain gravity using only short sentences.
### Input

### Output
""",

    # Edge case: empty input
    """### Instruction
Respond with a single word that means happy.
### Input

### Output
""",
]

for t in tests:
    infer(t)
