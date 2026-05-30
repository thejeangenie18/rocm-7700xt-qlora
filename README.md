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

## Screenshots (with DeafBlind-Standard Alt Text)

### Training Run
![Training run screenshot](images/training_run.png)
**Alt Text (DeafBlind Standard):**  
A terminal window showing a QLoRA training session for Qwen2.5‑3B.  
The screen displays progress bars, step counts, and loss values decreasing over time.  
Key metrics include:  
- trainable parameters around 29.9 million  
- total parameters around 3.1 billion  
- loss values trending downward (for example: 2.50 → 1.59 → 0.62 → 0.17)  
- training steps progressing from 0/200 to 200/200  
The terminal also shows ROCm warnings about hipBLAS fallback, which do not interrupt training.  
Overall, the screenshot communicates that the model is training successfully on AMD hardware.

### Model Output (Inference Test)
![Model output screenshot](images/model_output.png)
**Alt Text (DeafBlind Standard):**  
A Python REPL window showing the model loaded with the trained LoRA adapter.  
The user enters the prompt: “Explain spoon theory in simple terms.”  
The model responds with a clear, accessible explanation of Spoon Theory, describing spoons as units of energy used by people with chronic illness to manage daily tasks.  
The output demonstrates that the fine‑tuned model is generating coherent, disability‑aware responses.  
The screenshot confirms that the adapter loads correctly and inference works as expected.


## License
MIT for repository code.
Model weights follow their respective license terms.
