# Qwen2.5‑3B QLoRA Training (AMD ROCm + Quanto 4‑bit)

A lightweight, AMD-friendly QLoRA training pipeline for fine-tuning **Qwen/Qwen2.5-3B-Instruct** with **Quanto 4-bit quantization** and **LoRA adapters**. Built to run on 12GB VRAM GPUs such as the Radeon RX 7700 XT.

## Features
- ROCm-safe training (no bitsandbytes)
- Quanto 4-bit linear-only quantization
- LoRA with rslora enabled for Qwen2.5
- Works on 12GB VRAM
- Clean, line-delimited JSONL dataset format
- Reproducible training script

## Requirements
- AMD GPU with ROCm 6.x
- Python 3.10–3.11
- 12GB VRAM minimum
- 16GB RAM recommended

Install dependencies:

```bash
pip install transformers accelerate datasets peft quanto
```

## Dataset
Store training data in `data/train.jsonl`, one JSON object per line:

```json
{"instruction": "Explain spoon theory.", "input": "", "output": "Spoon theory is..."}
{"instruction": "Rewrite accessibly.", "input": "Original text", "output": "Accessible rewrite"}
```

## Training
Run the training script:

```bash
python scripts/train_qlora.py
```

On success, the run should show:
- trainable params ≈ 30M
- loss decreasing
- training steps progressing

The adapter is saved to:

```text
models/qwen25_3b_lora/adapter/
```

## Inference
Load the base model and apply the trained LoRA adapter:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "models/qwen25_3b_lora/adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)

prompt = "Explain spoon theory in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Merge LoRA into the Base Model

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "models/qwen25_3b_lora/adapter"
OUT = "models/qwen25_3b_merged"

model = AutoModelForCausalLM.from_pretrained(BASE, device_map="cpu")
model = PeftModel.from_pretrained(model, ADAPTER)
merged = model.merge_and_unload()
merged.save_pretrained(OUT)
```

## Screenshots
- `images/train.png`
- `images/output.png`

### Alt text
- Training run screenshot: terminal output showing progress bars, step counts, loss trending down, and ROCm warnings that do not interrupt training.
- Model output screenshot: Python REPL showing the trained adapter loaded successfully and a coherent explanation of Spoon Theory.

## License
MIT for repository code.
Model weights follow their respective license terms.
