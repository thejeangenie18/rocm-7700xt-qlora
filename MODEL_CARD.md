# Model Card — Qwen2.5‑3B QLoRA (4‑bit NF4, AMD ROCm RDNA3)

## Overview
This repository provides a **RDNA3‑safe QLoRA training and inference pipeline** for  
**Qwen/Qwen2.5‑3B‑Instruct**, using **4‑bit NF4 quantization** and **LoRA fine‑tuning** on  
consumer AMD GPUs (e.g., RX 7600 / 7700 XT / 7800 XT / 7900 XT/XTX).

The goal is to offer a **reproducible, local‑first workflow** for accessibility‑focused  
instruction tuning on ROCm 6.x hardware.

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
- ROCm 6.x
- 16GB system RAM recommended

### Software
- Python 3.10.X  
- `transformers`  
- `peft`  
- `datasets`  
- `bitsandbytes` (ROCm build)  

### Method
- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Quantization:** 4‑bit NF4 (linear layers only)
- **LoRA rank:** 64  
- **LoRA alpha:** 16  
- **LoRA dropout:** 0.05  
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj  
- **Optimizer:** `paged_adamw_32bit` (ROCm‑safe)
- **Sequence length:** 2048  
- **Batch size:** 1  
- **Gradient accumulation:** 8  
- **Dtype:** fp16  
- **RDNA3 stability settings:** expandable segments, SDMA enabled, TF32 matmul enabled

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

