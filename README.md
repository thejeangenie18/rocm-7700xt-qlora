# Qwen2.5‑3B QLoRA Training (AMD ROCm + 4‑bit NF4)

A fully RDNA3-safe QLoRA training pipeline for fine-tuning Qwen/Qwen2.5‑3B‑Instruct with 4-bit NF4 quantization and LoRA adapters.

Designed for local-first, reproducible training on 12GB VRAM AMD RDNA3 GPUs such as the Radeon RX 7700 XT.

## Features
- RDNA3-safe training with BitsAndBytes NF4 and fp16 compute
- ROCm-safe optimizer: `paged_adamw_32bit`
- LoRA rank 64 tuned for Qwen2.5
- 12GB VRAM friendly
- Clean JSONL dataset format
- CPU-safe LoRA merge into a single fp16 `safetensors` model
- RDNA3 stability settings included in all scripts

## Requirements
- AMD GPU with RDNA3 architecture (`gfx1100`, `gfx1101`, `gfx1102`)
- ROCm 6.1 or 6.2
- Python 3.10
- 12GB VRAM minimum
- 16GB RAM recommended

## System Requirements

This pipeline is tested on AMD RDNA3 hardware with ROCm 6.x. ROCm is sensitive to kernel and OS versions, so use a compatible setup.

### Supported Operating Systems
- Ubuntu 24.04 LTS (Noble) — recommended
- Ubuntu 22.04 LTS (Jammy) — supported with ROCm 6.x

### Recommended Kernel
- Kernel series: `6.8.x`
- Verified working kernels:
  - `6.8.0-49-generic`
  - `6.8.0-50-generic`

### ROCm Version
- Recommended: `6.1` or `6.2`
- `6.0` may be unstable on RDNA3
- `5.x` does not support RDNA3 GPUs

### GPU Support
- RDNA3: `gfx1100`, `gfx1101`, `gfx1102`
- Tested on: AMD Radeon RX 7700 XT (`gfx1101`)

## Verified Build Information

| Component      | Value                                |
|----------------|--------------------------------------|
| GPU            | AMD Radeon RX 7700 XT (`gfx1101`)    |
| ROCm Version   | `6.1`                                |
| OS             | Ubuntu 24.04.4 LTS (Noble)           |
| Kernel Version | `6.8.0-49-generic`                   |
| Python Version | `3.10.x`                             |
| PyTorch Build  | ROCm-enabled PyTorch (ROCm repo)     |
| VRAM           | `12 GB`                              |
| RAM            | `32 GB`                              |
| Storage        | NVMe SSD                             |
| Virtual Env    | `venv` (Python 3.10)                 |

## Installation

Follow the installation guide in `INSTALL.md` for the full ROCm + PyTorch + QLoRA environment on Ubuntu 24.04.

This includes:
- Python 3.10 setup for PyTorch ROCm wheels
- ROCm 6.1 installation for Ubuntu 24.04
- PyTorch ROCm wheel installation
- RDNA3 stability fixes from BEATEK_ROCm
- QLoRA dependencies
- Hugging Face CLI usage
- GPU inference verification

## Dataset Format

Store training data in `data/train.jsonl`, one JSON object per line:

```jsonl
{"instruction": "Explain spoon theory.", "input": "", "output": "Spoon theory is..."}
{"instruction": "Rewrite accessibly.", "input": "Original text", "output": "Accessible rewrite"}
```

Each example is converted to the following prompt template:

```
### Instruction
{instruction}
### Input
{input}
### Output
{output}
```

## Training

Run the RDNA3-safe training script:

```bash
python train_rdna3_fix.py
```

Expected signs of success:
- trainable params ≈ 29.9M
- loss decreasing
- training steps progressing from 0 to 200

The LoRA adapter is saved to:

`qwen3b_qlora_output/`

## Merge LoRA into a Single fp16 Model

After training, merge the adapter into a standalone fp16 model:

```bash
python merge_qwen_lora_final.py
```

Output:

```
qwen3b_merged_fp16_final/
  ├─ model.safetensors
  ├─ config.json
  └─ tokenizer.json
```

This merged model does not require PEFT and is ready for inference.

## Inference (Merged Model)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "qwen3b_merged_fp16_final"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},  # or "cuda:0" for GPU inference
    trust_remote_code=True,
)

prompt = "Explain spoon theory in simple terms."
inputs = tok(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(outputs[0], skip_special_tokens=True))
```

## RDNA3 Stability Credits

This project includes RDNA3-specific ROCm fixes originally documented by Beat-k in the BEATEK_ROCm project:
https://github.com/Beat-k/BEATEK_ROCm

## License

MIT for repository code.
Model weights follow their respective license terms.

