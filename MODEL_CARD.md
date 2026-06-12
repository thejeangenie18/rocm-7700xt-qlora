# Model Card — Qwen2.5‑3B QLoRA (RDNA3-Safe, ROCm 7.2.1)
## Overview  
This repository provides a RDNA3‑safe QLoRA training and inference pipeline for
Qwen/Qwen2.5‑3B‑Instruct, using LoRA fine‑tuning on consumer AMD GPUs
(e.g., RX 7600 / 7700 XT / 7800 XT / 7900 XT/XTX).

The pipeline is designed for local‑first, reproducible, accessibility‑focused instruction tuning without relying on CUDA‑only components.
All training and inference are performed using:  
- BF16 compute
- ROCm‑native kernels
- hipBLASLt → hipBLAS fallback
- SDMA‑enabled global memory ordering
- No Triton
- No FlashAttention
- No bitsandbytes

This model is the result of multiple validated training runs documented in training‑runs.md and analyzed in RESEARCH.md.
---

### RDNA3 Stability Credits
This project incorporates RDNA3‑specific ROCm fixes originally documented by
**Beat‑k** in the **BEATEK_ROCm** project (https://github.com/Beat-k/BEATEK_ROCm).
Their research into HSA overrides, SDMA enablement, allocator behavior, MFMA/WMMA hazards, waitcnt semantics, hipBLASLt fallback behavior and PyTorch initialization quirks directly informed the `.bashrc` environment settings used to stabilize QLoRA training on RDNA3 GPUs.

---

## Intended Use
### Recommended
- Local inference on AMD GPUs (RDNA3)
- Accessibility‑focused rewriting and simplification
- Assistive‑technology content generation
- Instruction following and structured reasoning
- Research on RDNA3‑safe QLoRA and BF16‑only matmul paths
- Community reproducibility studies

### Not Recommended
- Medical, legal, or financial decision‑making
- High‑stakes or safety‑critical applications
- Deployment without human oversight
- Use cases requiring guaranteed factual accuracy

---

## Training Details
### Hardware
- AMD RDNA3 GPU (validated on RX 7700 XT)
- ROCm 7.2.1
- 16GB system RAM recommended

### Software
- Python 3.10.X  
- `transformers`  
- `peft`  
- `datasets`  
- `accelerate`
- **No Triton, no bitsandbytes, no ONNX Runtime**

### Method
- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Quantization:** QLoRA 4‑bit adapters (NF4)
- **LoRA rank:** 64  
- **LoRA alpha:** 16  
- **LoRA dropout:** 0.05  
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj  
- **Optimizer:** `paged_adamw_32bit` (ROCm‑safe)
- **Sequence length:** 2048  
- **Batch size:** 1  
- **Gradient accumulation:** 8  
- **Dtype:** BF16 only
- RDNA3 stability settings:
  - `HSA_OVERRIDE_GFX_VERSION=11.0.0`
  - `HSA_ENABLE_SDMA=1`
  - `ROCM_FORCE_ENABLE_DP=1`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`

Training was performed using the RDNA3‑safe script:
`train_rdna3_fix.py`

All training runs (TinyLlama, Qwen3B incremental, Spoonie‑Helper v5) completed with:
- no hangs
- no FLAT_SCRATCH faults
- no WMMA corruption
- no masked‑lane artifacts
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
- `validate_examples.py`
- `schema.json`

---

## Model Merging
After training, the LoRA adapter is merged into a **single fp16 safetensors model** using:
- `merge_qwen_lora_final.py`

All merges were validated to be:
- EXEC‑stable
- free of masked‑lane divergence
- free of WMMA corruption
- free of Triton kernels
- compatible with CPU and GPU inference

The merged model is suitable for standalone inference without PEFT.
---

## Inference
Inference is performed using:
- `run_inference.py`
The script supports CPU and GPU inference and uses the same JSONL‑style prompt format as training.

---

## Evaluation
The model was evaluated manually using prompts related to:
- disability explanations  
- accessibility rewriting  
- general instruction following  
- edge‑case formatting (lists, emojis, short sentences)

Supports:
- CPU inference
- GPU inference (BF16)
- JSONL‑style prompt formatting

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
- Clear, plain‑language documentation  
- Screen‑reader‑friendly training logs  
- JSONL prompt format designed for readability  
- Consistent structure across training scripts  

---

## License
- Repository code: MIT  
- Base model: Qwen2.5 license (research use)  
- LoRA adapter and merged model: inherits base model license  

---

## Citation
If you use this work, please cite:
> Qwen2.5‑3B QLoRA (4‑bit NF4, AMD ROCm).  
> Local‑first fine‑tuning pipeline for accessibility‑focused AI.


---

### RDNA3 Stability Credits
This project incorporates RDNA3‑specific ROCm fixes originally documented by
**Beat‑k** in the **BEATEK_ROCm** project (https://github.com/Beat-k/BEATEK_ROCm).
Their research into HSA overrides, SDMA enablement, allocator behavior, and
PyTorch initialization quirks directly informed the `.bashrc` environment
settings used to stabilize QLoRA training on RDNA3 GPUs.

---

## Intended Use
### Recommended
- Local inference on AMD GPUs (RDNA3)
- Accessibility‑focused instruction following
- Assistive‑technology rewriting and simplification
- Research and experimentation with QLoRA on ROCm
- Educational and community projects

### Not Recommended
- Medical, legal, or financial decision‑making
- High‑stakes or safety‑critical applications
- Deployment without human oversight

---

## Training Details
### Hardware
- AMD RDNA3 GPU (validated on RX 7700 XT)
- ROCm 7.2.1
- 16GB system RAM recommended

### Software
- Python 3.10.X  
- `transformers`  
- `peft`  
- `datasets`  
- `accelerate`
- **No Triton, no bitsandbytes, no CUDA-only dependencies**

### Method
- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Quantization:** QLoRA 4‑bit adapters (NF4)
- **LoRA rank:** 64  
- **LoRA alpha:** 16  
- **LoRA dropout:** 0.05  
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj  
- **Optimizer:** `paged_adamw_32bit` (ROCm‑safe)
- **Sequence length:** 2048  
- **Batch size:** 1  
- **Gradient accumulation:** 8  
- **Dtype:** bf16/fp16  
- **RDNA3 stability settings:** HSA overrides, SDMA enabled, allocator tuning

Training was performed using `train_rdna3_fix.py`.

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
- `validate_examples.py`
- `schema.json`

---

## Model Merging
After training, the LoRA adapter is merged into a **single fp16 safetensors model** using:
- `merge_qwen_lora_final.py`
This produces a standalone model suitable for inference without PEFT.

---

## Inference
Inference is performed using:
- `run_inference.py`
The script supports CPU and GPU inference and uses the same JSONL‑style prompt format as training.

---

## Evaluation
The model was evaluated manually using prompts related to:
- disability explanations  
- accessibility rewriting  
- general instruction following  
- edge‑case formatting (lists, emojis, short sentences)

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
- Clear, plain‑language documentation  
- JSONL prompt format designed for readability  
- Scripts validated with screen‑reader‑friendly output  

---

## License
- Repository code: MIT  
- Base model: Qwen2.5 license (research use)  
- LoRA adapter and merged model: inherits base model license  

---

## Citation
If you use this work, please cite:
> Qwen2.5‑3B QLoRA (4‑bit NF4, AMD ROCm).  
> Local‑first fine‑tuning pipeline for accessibility‑focused AI.

