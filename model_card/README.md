# Model Card: RDNA3-Safe QLoRA Pipeline (ROCm 7.2.4)

## Overview

This repository provides an RDNA3-safe QLoRA training and inference pipeline for
any HuggingFace causal language model, using LoRA fine-tuning on consumer AMD GPUs
(e.g., RX 7600 / 7700 XT / 7800 XT / 7900 XT/XTX).

The pipeline is designed for local-first, reproducible, accessibility-focused instruction tuning without relying on CUDA-only components.
All training and inference are performed using:
- BF16 compute
- ROCm-native kernels
- hipBLASLt with hipBLAS fallback
- SDMA-enabled global memory ordering
- No Triton
- No FlashAttention
- No bitsandbytes

Results are documented across multiple validated training runs (TinyLlama, Qwen2.5-3B incremental, Spoonie-Helper v5).

---

### RDNA3 Stability Credits
This project incorporates RDNA3-specific ROCm fixes originally documented by
**Beat-k** in the [**BEATEK_ROCm** project] (https://github.com/Beat-k/BEATEK_ROCm).  
Their research into HSA overrides, SDMA enablement, allocator behavior, MFMA/WMMA hazards, waitcnt semantics, hipBLASLt fallback behavior, and PyTorch initialization quirks directly informed the `.bashrc` environment settings used to stabilize QLoRA training on RDNA3 GPUs.

---

Unified RDNA3 + Zen 3 Research Repository

This model card documents the training and inference pipeline.
All architectural findings, kernel‑level analysis, Zen 3 CPU stability requirements, and reproducibility logs that informed this pipeline are maintained in a dedicated research repository:  
[RDNA3 ZEN3 Unified Research Repo](https://github.com/thejeangenie18/rdna3-zen3-unified)


The unified repository includes:  
- RDNA3 kernel behavior analysis
- Zen 3 CPU stability requirements
- SDMA and hipBLASLt behavior notes
- WMMA tile hazards and LDS visibility findings
- Memory‑model inconsistencies and mitigation
- Full fingerprint matrix
- Structured stability reports
- Accessibility‑structured ingestion datasets
- Cross‑platform reproducibility logs

This QLoRA pipeline implements the stable subset of those findings.  
The unified repository explains the architectural reasons behind each stability rule.

---

## Intended Use
### Recommended
- Local inference on AMD GPUs (RDNA3)
- Accessibility-focused rewriting and simplification
- Assistive-technology content generation
- Instruction following and structured reasoning
- Research on RDNA3-safe QLoRA and BF16-only matmul paths
- Community reproducibility studies

### Not Recommended
- Medical, legal, or financial decision-making
- High-stakes or safety-critical applications
- Deployment without human oversight
- Use cases requiring guaranteed factual accuracy

---

## Training Details
### Hardware
- AMD RDNA3 GPU (validated on RX 7700 XT)
- ROCm 7.2.4
- 16GB system RAM recommended

### Software
- Python 3.10.x
- `transformers`
- `peft`
- `datasets`
- `accelerate`
- **No Triton, no bitsandbytes, no ONNX Runtime**

### Method
- **Base model:** configurable; any HuggingFace causal LM (validated on Qwen2.5-3B-Instruct and TinyLlama-1.1B)
- **Quantization:** QLoRA 4-bit adapters (NF4)
- **LoRA rank:** 64 (default; configurable)
- **LoRA alpha:** 16
- **LoRA dropout:** 0.05
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (Llama/Qwen family defaults; adjust for other architectures)
- **Optimizer:** `paged_adamw_32bit` (ROCm-safe)
- **Sequence length:** 2048
- **Batch size:** 1
- **Gradient accumulation:** 8
- **Dtype:** BF16 only
- RDNA3 stability settings:
  - `HSA_OVERRIDE_GFX_VERSION=11.0.0`
  - `HSA_ENABLE_SDMA=1`
  - `ROCM_FORCE_ENABLE_DP=1`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`

Training was performed using the RDNA3-safe script:
`scripts/train_rdna3_fix.py`

All training runs (TinyLlama, Qwen2.5-3B incremental, Spoonie-Helper v5) completed with:
- no hangs
- no FLAT_SCRATCH faults
- no WMMA corruption
- no masked-lane artifacts
- no Triton kernels
- stable BF16 gradients

---

## Dataset Format
Training data is stored in JSONL format:
```json
{"instruction": "...", "input": "...", "output": "..."}
```

The training script converts each example into a structured prompt:
```
### Instruction
{instruction}
### Input
{input}
### Output
{output}
```

Dataset validation tools are included in:
- `scripts/validate_adapter.py`
- `schema/schema.json`

---

## Model Merging
After training, the LoRA adapter can be validated and merged into a **single fp16 safetensors model** using:
- `scripts/merge_adapter_final.py`

All merges were validated to be:
- EXEC-stable
- free of masked-lane divergence
- free of WMMA corruption
- free of Triton kernels
- compatible with CPU and GPU inference

The merged model is suitable for standalone inference without PEFT.

---

## Inference
Inference is performed using:
- `scripts/run_inference.py`

The script supports CPU and GPU inference and uses the same JSONL-style prompt format as training.

---

## Evaluation
The model was evaluated manually using prompts related to:
- disability explanations
- accessibility rewriting
- general instruction following
- edge-case formatting (lists, emojis, short sentences)

Supports:
- CPU inference
- GPU inference (BF16)
- JSONL-style prompt formatting

Example:
```
Explain spoon theory in simple terms.
```

Outputs were coherent, accessible, and consistent with the training objectives.

---

## Ethical Considerations
- The model may generate incorrect or outdated information.
- Outputs may reflect biases present in the base model or dataset.
- Human review is required for sensitive or consequential use cases.
- This model is not a substitute for professional expertise.

---

## Accessibility Notes
This repository prioritizes accessibility:
- Alt text for screenshots
- Clear, plain-language documentation
- Screen-reader-friendly training logs
- JSONL prompt format designed for readability
- Consistent structure across training scripts

---

## License
- Repository code: MIT
- Base model: refer to the chosen model's license terms
- LoRA adapter: inherits base model license

---

## Citation
If you use this work, please cite:
> RDNA3-Safe QLoRA Pipeline (AMD ROCm).
> Local-first fine-tuning pipeline for accessibility-focused AI.
