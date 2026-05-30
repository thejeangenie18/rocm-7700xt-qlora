# Model Card — Qwen2.5‑3B QLoRA (Quanto 4‑bit, AMD ROCm)

## Overview
This repository contains a fine-tuned adapter for **Qwen/Qwen2.5-3B-Instruct** trained with **QLoRA** and **Quanto 4-bit linear-only quantization** on **AMD ROCm 6.x** hardware. It is designed for reproducible local training and inference on 12GB VRAM consumer GPUs.

## Intended use
### Recommended
- Local inference on AMD GPUs
- Accessibility-focused instruction following
- Research and experimentation with QLoRA on ROCm
- Educational use

### Not recommended
- Medical diagnosis
- Legal advice
- High-stakes decision-making
- Deployment without human oversight

## Training
### Hardware
- AMD GPU (RDNA3 recommended)
- ROCm 6.x
- 16GB RAM recommended

### Software
- Python 3.10–3.11
- transformers
- accelerate
- peft
- quanto
- datasets

### Method
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Quantization: Quanto `qint4` (linear layers only)
- LoRA rank: 16
- LoRA dropout: 0.05
- rslora: enabled
- Sequence length: 384
- Batch size: 1
- Gradient accumulation: 8
- Optimizer: `adamw_torch` (ROCm-safe)
- Dtype: bf16 if supported, otherwise fp16

## Dataset
Training data is stored in `data/train.jsonl`, with one JSON object per line:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

## Evaluation
The model was evaluated manually using prompts focused on:
- disability explanations
- accessibility rewriting
- general instruction following

Example prompt:

```text
Explain spoon theory in simple terms.
```

Evaluation showed that the LoRA adapter loads correctly and produces accessible, coherent responses.

## Ethical considerations
- The model may generate incorrect or outdated information
- The model may reflect biases in the base model or dataset
- Human review is required for sensitive use cases
- This model is not a substitute for professional expertise

## Accessibility
- Alt text is provided for screenshots
- Training and inference examples are described in clear language
- Screenshots are stored in the `images/` directory and referenced from `README.md`

## License
- Repository code: MIT
- Base model: Qwen2.5 license / research use
- LoRA adapter: follows the base model license terms

## Citation
If you use this work, please cite:

> Qwen2.5‑3B QLoRA (Quanto 4‑bit, AMD ROCm). Local-first fine-tuning pipeline for accessibility-focused AI.



