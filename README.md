# RDNA3 QLoRA Training Pipeline (AMD ROCm)

A fully RDNA3-safe QLoRA training pipeline for fine-tuning any HuggingFace causal language model using LoRA adapters on AMD RDNA3 GPUs.  
This project is designed for local-first, reproducible training on 12GB VRAM RDNA3 GPUs such as the Radeon RX 7700 XT.  
This pipeline uses no Triton, no BitsAndBytes, and no CUDA-only kernels. Everything is validated on ROCm 7.2.4.  

---

## Features

- RDNA3-safe QLoRA training (no Triton, no bitsandbytes)
- ROCm-safe optimizer (paged_adamw_32bit)
- Configurable LoRA rank (default 64)
- 12GB VRAM friendly (fits 3B models comfortably)
- Clean JSONL dataset format
- LoRA adapter validation for merge safety (no actual merging)
- RDNA3 stability settings included in all scripts
- Minimal, stable ROCm 7.2.4 Python environment
- Incremental training support
- System snapshots (ROCm SMI + system stats)
- Verified BF16-only matmul path
- Verified hipBLASLt fallback behavior
- Verified SDMA-mediated stability
- Verified merge-safe LoRA adapters (TinyLlama, Qwen2.5-3B)

---

## Ongoing Findings (ROCm 7.2.4)

ROCm 7.2.4 is the current recommended version for RDNA3 training on this pipeline.
It introduces kernel-correctness improvements for RDNA3 over the 7.2.0 to 7.2.3 range, including improved WMMA tile handling. The RDNA3 stability environment variables documented here remain precautionary and are still recommended on 7.2.4.

Earlier findings documenting pre-fix behavior (ROCm 7.2.0 to 7.2.3) are preserved in the separate research repository. New findings for 7.2.4 will be added as additional training runs, inference traces, and reproducibility reports are collected. Contributions and independent logs from other RDNA3 users are welcome.

---

## Requirements

- AMD GPU with RDNA3 architecture (`gfx1100`, `gfx1101`, `gfx1102`)
- ROCm 7.2.4 (recommended); 7.2.1+ supported
- Python 3.10
- 12GB VRAM minimum
- 16GB RAM recommended

---

## System Requirements

This pipeline is tested on ROCm 7.2.4 with AMD RDNA3 hardware.
ROCm is sensitive to kernel and OS versions, so use a compatible setup.

---

### Supported Operating Systems

- Ubuntu 24.04 LTS (Noble): recommended

---

### Recommended Kernel

- Kernel series: `6.8.x`
- Verified working kernels:
  - `6.8.0-49-generic`
  - `6.8.0-50-generic`

---

### ROCm Version

- Recommended: `7.2.4`
- `6.x` is older and less stable on RDNA3
- `5.x` does not support RDNA3 GPUs

---

### GPU Support

- RDNA3: `gfx1100`, `gfx1101`, `gfx1102`
- Tested on: AMD Radeon RX 7700 XT (`gfx1101`)

---

## Verified Build Information

| **Component**  | **Value**                            |
|----------------|--------------------------------------|
| GPU            | AMD Radeon RX 7700 XT (`gfx1101`)    |
| ROCm Version   | `7.2.4`                              |
| OS             | Ubuntu 24.04.4 LTS (Noble)           |
| Kernel Version | `6.8.0-49-generic`                   |
| Python Version | `3.10.x`                             |
| PyTorch Build  | ROCm-enabled PyTorch (ROCm 7.2)      |
| VRAM           | `12 GB`                              |
| RAM            | `32 GB`                              |
| Storage        | NVMe SSD                             |
| Virtual Env    | `venv` (Python 3.10)                 |

---

## Installation
See the full setup guide in [install/README.md](install/README.md).

This covers:  
- Python 3.10 environment setup
- ROCm 7.2.4 installation for Ubuntu 24.04
- PyTorch ROCm 7.2 wheels
- RDNA3 stability fixes
- QLoRA dependencies
- GPU inference verification

---

## Dataset Format

Store training data in `datasets/train.jsonl`, one JSON object per line:

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

---

## Training

Run the RDNA3-safe training script:

```bash
python scripts/train_rdna3_fix.py
```

Configure the model, dataset, and output paths via environment variables:

```bash
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
DATA_PATH=./datasets/ada.jsonl \
OUTPUT_DIR=./loras/my-adapter \
python scripts/train_rdna3_fix.py
```

Expected signs of success:  
- trainable params visible in output
- loss decreasing
- BF16 compute
- no Triton kernels
- hipBLASLt fallback warning
- no FLAT_SCRATCH faults
- no hangs

The LoRA adapter is saved to the configured `OUTPUT_DIR` (default: `./loras/adapter`).

**Additional Verified Training Pipelines:**  
- TinyLlama (314 examples): stable 3-epoch run
- Qwen2.5-3B incremental training: stable 2208-step run
- Spoonie-Helper v5: stable 11,764-example run

All runs produce:  
- stable gradients
- no corruption
- no deadlocks
- correct merges

---

## Merge LoRA into a Single fp16 Model

After training, validate and optionally merge the adapter:

```bash
python scripts/merge_adapter_final.py \
  --base_model <owner>/<model> \
  --adapter ./loras/my-adapter \
  --output ./models/merged
```

This runs merge-safety validation. The merged model does not require PEFT and is ready for inference.

---

## Inference (Merged Model)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "./models/merged"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},  # or "cuda:0" for GPU inference (ROCm maps to this namespace)
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
- No masked-lane artifacts
- No WMMA corruption
- No Triton kernels involved

---

## CPU Environment Requirements (Zen 3 / Ryzen 5700X3D)

Modern RDNA3 training stability depends not only on GPU-side fixes but also on a correct Zen 3 CPU environment.
On Ryzen 5700X3D systems, several CPU-side power-management features can cause training instability, early-epoch crashes, or dataloader stalls.
This project includes a validated configuration that eliminates those issues.

**Verified CPU-side stability requirements:**  
- Disable Global C-States (prevents frequency-collapse stalls)
- Disable PCIe ASPM (removes link-state latency that disrupts SDMA)
- Enable IOMMU + SVM (required for ROCm correctness)
- Kernel parameters: `amd_iommu=on iommu=pt pcie_aspm=off processor.max_cstate=1`
- Runtime Settings
    - CPU Governor = performance
    - EPP = performance
    - THP = madvise

These fixes are validated across all QLoRA training runs and are required for stable multi-epoch training on Zen 3 hardware.

---

## Scripts Reference

| **Script** | **Purpose** |
|--------|---------|
| `scripts/train_rdna3_fix.py` | RDNA3-safe QLoRA training; configure via environment variables |
| `scripts/merge_adapter_final.py` | CLI-driven merge-safety validation for trained LoRA adapters |
| `scripts/merge_adapter_config.py` | Config-file-driven merge validation |
| `scripts/merge_adapter.py` | Minimal merge validation with argparse |
| `scripts/run_inference.py` | Interactive inference with a merged model |
| `scripts/eval.py` | Evaluation suite: load base and adapter, run test prompts |
| `scripts/validate_adapter.py` | Run categorized prompts and log outputs to JSONL |
| `scripts/freeze_adapter.py` | Snapshot adapter files to a versioned release directory |
| `scripts/chain_train.py` | Multi-stage TRAIN to VALIDATE pipeline runner |
| `tools/smoke_test.py` | Quick forward/backward smoke test for any model or adapter |
| `tools/validate_all.py` | Full validation suite: forward pass, KL divergence, shape checks |

---

## RDNA3 Stability Credits

This project includes RDNA3-specific ROCm fixes originally documented by Beat-k in the BEATEK_ROCm project:  
[BEATEK_ROCm](https://github.com/Beat-k/BEATEK_ROCm)



## Research Impact and Further Context

The RDNA3 kernel-level findings documented in this repository, including silent numerical corruption, WMMA dependency hazards, LDS visibility issues, and memory-model inconsistencies, directly informed a public article on AI brittleness in healthcare. The same architectural patterns that cause silent failures in ML kernels also appear in real clinical AI workflows, where they manifest as inaccessible, unreliable, or unsafe system behavior.

---

## Unified Research Repository: RDNA3 + Zen 3 Stability Work

This QLoRA pipeline was built from a larger stability research project that documented RDNA3 GPU behavior, Zen 3 CPU interaction patterns, and the architectural causes of silent numerical corruption on AMD hardware.  

The full research corpus, including the fingerprint matrix and the structured stability taxonomy, now lives in a dedicated repository: [RDNA3 ZEN3 Unified Research Repo](https://github.com/thejeangenie18/rdna3-zen3-unified)

The unified repository contains:  
- RDNA3 kernel‑level findings
- Zen 3 CPU stability requirements
- SDMA and hipBLASLt fallback behavior
- WMMA tile hazards and LDS visibility notes
- Memory‑model inconsistencies
- Full fingerprint matrix
- Structured stability reports
- Accessibility‑structured ingestion datasets
- Cross‑platform reproducibility logs

This LoRA pipeline implements the stable subset of those findings.  
The unified repository explains the architectural reasons behind each stability rule.  

---

## License
MIT for repository code.
Model weights follow their respective license terms.
