#!/usr/bin/env python3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# RDNA3 / ROCm STABILITY SETTINGS
# -----------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HSA_ENABLE_SDMA"] = "1"
os.environ["ROCM_FORCE_ENABLE_DP"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------
# CONFIG
# -----------------------------
MODEL = "/home/jg18/Project/rocm-7700xt-qlora/qwen3b_merged_fp16_final"
USE_GPU = False  # set True for GPU inference

device = "cuda:0" if USE_GPU else "cpu"

print("Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading merged model on {device} (bf16 compute)…")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,   # RDNA3-native
    device_map={"": device},
    trust_remote_code=True,
)

model.eval()

# -----------------------------
# PROMPT FORMATTER (JSONL STYLE)
# -----------------------------
def format_instruction(instruction: str) -> str:
    escaped = instruction.replace('"', '\\"')
    return f'{{"instruction": "{escaped}", "input": "", "output": "'


# -----------------------------
# CHAT FUNCTION
# -----------------------------
def chat(prompt: str, max_new_tokens: int = 256) -> str:
    text = format_instruction(prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the model's output field
    if '"output": "' in decoded:
        decoded = decoded.split('"output": "', 1)[-1]

    if decoded.endswith('"'):
        decoded = decoded[:-1]

    return decoded.strip()


# -----------------------------
# INTERACTIVE LOOP
# -----------------------------
if __name__ == "__main__":
    print("Merged instruction-tuned model ready (JSONL format).\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        print("\nAssistant:", chat(q), "\n")
