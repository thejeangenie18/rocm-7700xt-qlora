import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

model = PeftModel.from_pretrained(
    base,
    "releases/tinyllama-qlora/final",
    torch_dtype=torch.float16
)

def infer(prompt):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0
    )
    print(tok.decode(out[0], skip_special_tokens=True))
    print("-" * 80)

tests = [
    # SQL-only enforcement (correct format)
    """### Instruction
Respond only with SQL.
### Input
List all spoonie-owned stores.
### Output
""",

    # Structured query
    "### Instruction\nExecute structured query.\n### Input\n{\"action\":\"query\",\"filter\":{\"vendor\":\"Target\"}}\n### Output",

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
    "### Instruction\nExecute structured query.\n### Input\n{}\n### Output"
]

for t in tests:
    infer(t)

