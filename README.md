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

- ## System Requirements
This pipeline is tested and verified on AMD RDNA3 hardware using ROCm 6.x.  
ROCm is sensitive to kernel and OS versions, so matching these is important for reproducibility.

### Supported Operating Systems
- Ubuntu 24.04 LTS (Noble) — recommended
- Ubuntu 22.04 LTS (Jammy) — supported with ROCm 6.x
- Other distros: not officially supported by AMD for ROCm

### Required Kernel Versions
ROCm 6.x requires a kernel in the **6.8.x** series for stable RDNA3 support.

Verified working kernels:
- `6.8.0-49-generic`
- `6.8.0-50-generic`

Not recommended:
- 7.x kernels (ROCm DKMS modules fail to build)
- 5.x kernels (missing RDNA3 support)

### Required ROCm Version
- ROCm **6.1** or **6.2** recommended
- ROCm 6.0 works but has instability with RDNA3
- ROCm 5.x does *not* support RDNA3 GPUs

### GPU Support
- RDNA3 (gfx1100, gfx1101, gfx1102)
- Tested specifically on **Radeon RX 7700 XT (gfx1101)**

## Verified Build Information
| Component        | Value                                   |
|------------------|-------------------------------------------|
| GPU              | AMD Radeon RX 7700 XT (gfx1101)          |
| ROCm Version     | 6.1                                       |
| OS               | Ubuntu 24.04.4 LTS (Noble)                |
| Kernel Version   | 6.8.0-49-generic                          |
| Python Version   | 3.11.x                                    |
| PyTorch Build    | ROCm-enabled PyTorch (from rocm repo)     |
| VRAM             | 12 GB                                     |
| RAM              | 32 GB                                     |
| Storage          | NVMe SSD                                  |
| Virtual Env      | venv (Python 3.11)                        |

---

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
![Training run screenshot](images/train.png)
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
![Model output screenshot](images/output.png)
**Alt Text (DeafBlind Standard):**  
A Python REPL window showing the model loaded with the trained LoRA adapter.  
The user enters the prompt: “Explain spoon theory in simple terms.”  
The model responds with a clear, accessible explanation of Spoon Theory, describing spoons as units of energy used by people with chronic illness to manage daily tasks.  
The output demonstrates that the fine‑tuned model is generating coherent, disability‑aware responses.  
The screenshot confirms that the adapter loads correctly and inference works as expected.


## License
MIT for repository code.
Model weights follow their respective license terms.
