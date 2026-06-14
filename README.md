# Qwen2.5‑3B QLoRA Training (AMD ROCm + 4‑bit Quanto)
A fully RDNA3‑safe QLoRA training pipeline for fine‑tuning Qwen/Qwen2.5‑3B‑Instruct using LoRA adapters on AMD GPUs.
This project is designed for local‑first, reproducible training on 12GB VRAM RDNA3 GPUs such as the Radeon RX 7700 XT.
This pipeline uses no Triton, no BitsAndBytes, and no CUDA‑only kernels. Everything is validated on ROCm 7.2.1.

## Features
- RDNA3‑safe QLoRA training (no Triton, no bitsandbytes)
- ROCm‑safe optimizer (paged_adamw_32bit)
- LoRA rank 64 tuned for Qwen2.5
- 12GB VRAM friendly (fits 3B models comfortably)
- Clean JSONL dataset format
- LoRA adapter validation for merge safety (no actual merging)
- RDNA3 stability settings included in all scripts
- Minimal, stable ROCm 7.2.1 Python environment
- Incremental training support (Qwen3B incremental run)
- System snapshots (ROCm SMI + system stats)
- Verified BF16‑only matmul path
- Verified hipBLASLt fallback behavior
- Verified SDMA‑mediated stability
- Verified merge-safe LoRA adapters (TinyLlama, Qwen3B)

## Ongoing Findings (ROCm 7.2.4+)
ROCm 7.2.4 introduces several kernel‑correctness improvements for RDNA3.
The research in this repository currently documents pre‑fix behavior
(ROCm 7.2.0–7.2.3), and new findings for 7.2.4+ will be added as the
project develops.

As additional QLoRA training runs, inference traces, and reproducibility
reports are collected, the post‑fix section of the research will expand.
Contributions and independent logs from other RDNA3 users are welcome.

## Requirements
- AMD GPU with RDNA3 architecture (`gfx1100`, `gfx1101`, `gfx1102`)
- ROCm 7.2.1
- Python 3.10
- 12GB VRAM minimum
- 16GB RAM recommended

## System Requirements
This pipeline is tested on ROCm 7.2.1 with AMD RDNA3 hardware.
ROCm is sensitive to kernel and OS versions, so use a compatible setup.

### Supported Operating Systems
- Ubuntu 24.04 LTS (Noble) — recommended

### Recommended Kernel
- Kernel series: `6.8.x`
- Verified working kernels:
  - `6.8.0-49-generic`
  - `6.8.0-50-generic`

### ROCm Version
- Recommended: `7.2.1`
- `6.x` is older and less stable on RDNA3
- `5.x` does not support RDNA3 GPUs

### GPU Support
- RDNA3: `gfx1100`, `gfx1101`, `gfx1102`
- Tested on: AMD Radeon RX 7700 XT (`gfx1101`)

## Verified Build Information
| Component      | Value                                |
|----------------|--------------------------------------|
| GPU            | AMD Radeon RX 7700 XT (`gfx1101`)    |
| ROCm Version   | `7.2.1`                              |
| OS             | Ubuntu 24.04.4 LTS (Noble)           |
| Kernel Version | `6.8.0-49-generic`                   |
| Python Version | `3.10.x`                             |
| PyTorch Build  | ROCm-enabled PyTorch (ROCm 7.2)      |
| VRAM           | `12 GB`                              |
| RAM            | `32 GB`                              |
| Storage        | NVMe SSD                             |
| Virtual Env    | `venv` (Python 3.10)                 |

## Installation
See the full setup guide in [INSTALL.md](INSTALL.md).

This covers:
- Python 3.10 environment setup
- ROCm 7.2.1 installation for Ubuntu 24.04
- PyTorch ROCm 7.2 wheels
- RDNA3 stability fixes
- QLoRA dependencies
- GPU inference verification

For RDNA3 ISA notes, Triton hazards, and kernel‑level analysis, see [RESEARCH.md](research/README.md).

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
- BF16 compute
- no Triton kernels
- hipBLASLt fallback warning
- no FLAT_SCRATCH faults
- no hangs

The LoRA adapter is saved to:

`qwen3b_qlora_output/`

**Additional Verified Training Pipelines:**  
- TinyLlama (314 examples) — stable 3‑epoch run
- Qwen3B incremental training — stable 2208‑step run
- Spoonie‑Helper v5 — stable 11,764‑example run

All runs produce:  
- stable gradients
- no corruption
- no deadlocks
- correct merges

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

**Inference Stability Guarantees**
- BF16 GPU inference is stable 
- No EXEC divergence 
- No masked‑lane artifacts 
- No WMMA corruption 
- No Triton kernels involved

## CPU Environment Requirements (Zen 3 / Ryzen 5700X3D)
Modern RDNA3 training stability depends not only on GPU‑side fixes but also on a correct Zen 3 CPU environment.  
On Ryzen 5700X3D systems, several CPU‑side power‑management features can cause training instability, early‑epoch crashes, or dataloader stalls.  
This project includes a validated configuration that eliminates those issues.

**Verified CPU‑side stability requirements:**
- Disable Global C‑States (prevents frequency‑collapse stalls)
- Disable PCIe ASPM (removes link‑state latency that disrupts SDMA)
- Enable IOMMU + SVM (required for ROCm correctness)
- Kernel parameters: `amd_iommu=on iommu=pt pcie_aspm=off processor.max_cstate=1` 
- Runtime Settings
    - CPU Governor = performance
    - EPP = performance
    - THP = madvise
    
These fixes are now validated across all QLoRA runs and are required for stable multi‑epoch training on Zen 3 hardware.  
For full details, see the expanded section in [RESEARCH.md](research/README.md#10-zen-3--ryzen-5700x3d-cpu-environmental-fixes)

## RDNA3 Stability Credits
This project includes RDNA3-specific ROCm fixes originally documented by Beat-k in the BEATEK_ROCm project:
https://github.com/Beat-k/BEATEK_ROCm

## License
MIT for repository code.
Model weights follow their respective license terms.
