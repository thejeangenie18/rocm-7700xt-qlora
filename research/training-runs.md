# Training Runs — QLoRA on RDNA3 (7700 XT)

This document tracks all reproducible training runs performed on the ROCm‑enabled AMD Radeon 7700 XT system.  
Each run includes:  
- Model  
- Dataset  
- TrainingArguments  
- Loss metrics  
- Hardware telemetry (ROCm SMI + system stats)  
- Notes on stability, hazards, and performance  

See **Appendix A** for the list of models currently deployed in production.

---

# Run 1 — Spoonie Helper v5 (Qwen2.5‑3B, expanded_MASTER.jsonl, 10k)

### **Run Summary**
| Field | Value |
|-------|--------|
| Date | 2026‑06‑10 |
| Model | SpoonieHelper‑v5 |
| Base Model | Qwen2.5‑3B‑Instruct |
| Dataset | expanded_MASTER.jsonl |
| Dataset Size | 10,000 |
| Epochs | 1 |
| Total Steps | 669 |
| Loss Start | 4.1868 |
| Loss End | 1.176978161873126 |
| Loss Delta | 3.0098 |
| Runtime | 42.78 min |
| Steps/sec | 0.261 |
| Tokens/sec | ~8540 |
| VRAM Peak | ~9.7 GB (76%) |
| ROCm Version | 7.2.1 |
| Notes | Stable run, no hazards, smooth LR decay |

### **TrainingArguments**
```
num_train_epochs=1
per_device_train_batch_size=1
gradient_accumulation_steps=16
learning_rate=1.5e-4
bf16=True
logging_steps=10
save_steps=250
optim="adamw_torch"
gradient_checkpointing=True
remove_unused_columns=False
```

### **Hardware Telemetry**
**Pre‑run ROCm SMI**
- Temp: 39°C  
- Power: 24W  
- VRAM: 6%  

**Post‑run ROCm SMI**
- Temp: 63°C  
- Power: 118W  
- VRAM: 76%  

**System Stats**
- RAM: 2.6 GiB → 4.4 GiB  
- Swap: unchanged  
- IO Wait: 0%  

---

# Run 2 — TinyLlama‑AT‑v4 (1.1B, tinyllama-big.jsonl, 1480)

### **Run Summary**
| Field | Value |
|-------|--------|
| Date | 2026‑06‑10 |
| Model | TinyLlama‑AT‑v4 |
| Base Model | TinyLlama‑1.1B‑Chat |
| Dataset | tinyllama-big.jsonl |
| Dataset Size | 1480 |
| Epochs | 2 |
| Total Steps | 92 |
| Loss Start | 3.5881 |
| Loss End | 2.0848 |
| Loss Delta | 1.5033 |
| Runtime | 7.23 min |
| Steps/sec | 0.212 |
| Tokens/sec | ~4340 |
| VRAM Peak | ~4.5 GB (35%) |
| ROCm Version | 7.2.1 |
| Notes | Very stable, low VRAM footprint, ideal for Pi deployment |

### **TrainingArguments**
```
num_train_epochs=2
per_device_train_batch_size=1
gradient_accumulation_steps=16
learning_rate=2e-4
bf16=True
logging_steps=10
save_steps=200
optim="adamw_torch"
gradient_checkpointing=True
remove_unused_columns=False
```

### **Hardware Telemetry**
**Pre‑run ROCm SMI**
- Temp: 31°C  
- Power: 6W  
- VRAM: 28%  

**Post‑run ROCm SMI**
- Temp: 51°C  
- Power: 96W  
- VRAM: 35%  

**System Stats**
- RAM: 4.2 GiB → 4.7 GiB  
- Swap: unchanged  
- IO Wait: 0%  

---

# Run 3 — Spoonie Helper v4 (Qwen2.5‑3B, MASTER.jsonl, 2,679)

### **Run Summary**
| Field | Value |
|-------|--------|
| Base Model | Qwen2.5‑3B‑Instruct |
| Dataset | MASTER.jsonl |
| Dataset Size | 2,679 |
| Epochs | 1.2 |
| Total Steps | 167 |
| Loss Start | 4.2309 |
| Loss End | 2.0678 |
| Loss Delta | –2.1631 |
| Runtime | 10.31 min |
| Steps/sec | 0.27 |
| GTT Used | 0.41 GB |
| ROCm Version | 7.2.1 |
| Notes | Stable run, no hazards, smooth curve, no warmup spike |

### **TrainingArguments**
```
num_train_epochs=1.2
per_device_train_batch_size=1
gradient_accumulation_steps=32
learning_rate=1e-4
bf16=True
logging_steps=10
save_steps=250
optim="adamw_torch"
gradient_checkpointing=True
remove_unused_columns=False
```

### **Hardware Telemetry**
*(VRAM, UMA, CPU RAM, IO Wait not recorded for this run)*

# Run 4 - TinyLlama‑AT‑v5 (1.1B, tinyllama‑ada.jsonl, 314 examples)

### Run Summary
| Field | Value |
|-------|--------|
| Date	| 2026‑06‑12 |
| Model	| TinyLlama‑AT‑v5 |
| Base Model | TinyLlama‑1.1B‑Chat |
| Dataset | tinyllama‑314.jsonl |
| Dataset Size | 314 | 
| Epochs | 3 |
| Total Steps | 18
| Loss Start | 2.005 |
| Loss End | 0.3635 |
| Loss Delta | –1.6415 |
| Runtime | 101.6 sec |
| Steps/sec | 0.591 |
| Samples/sec | 9.275 |
| VRAM Peak | ~4.3 GB |
| ROCm Version | 7.2.1 |
| Notes | Extremely stable; ideal for rapid iteration and adapter prototyping |

**TrainingArguments** 
```
num_train_epochs=3
per_device_train_batch_size=1
gradient_accumulation_steps=16
learning_rate=1.75e-4
bf16=True
logging_steps=10
save_steps=200
optim="adamw_torch"
gradient_checkpointing=True
remove_unused_columns=False
```

**Hardware Telemetry**
Pre-run: 
- Temp: 33°C
- Power: 7W
- VRAM: 28%

Post‑run:  
- Temp: 49°C
- Power: 92W
- VRAM: 34%  

System RAM
- 4.1 GiB → 4.5 GiB

# Run 5 - Qwen2.5-3B Incremental (ada.jsonl, 11,764 examples, 3 epochs)

### Run Summary
| Field | Value |
|-------|--------|
| Date | 2026‑06‑12 |
| Model | Qwen3B‑Incremental |
| Base Model | Qwen2.5‑3B‑Instruct |
| Dataset | ada.jsonl |
| Dataset Size | 11,764 |
| Epochs | 3 |
| Total Steps | 2,208 |
| Loss Start | ~3.2 |
| Loss End | ~0.19 |
| Loss Delta | –3.0 |
| Runtime | 1 hr 49 min |
| Steps/sec | 0.337 |
| Samples/sec |5.388 |
| VRAM Peak | ~9.8 GB |
| ROCm Version |7.2.1 |
| Notes | Long‑run stability confirmed; no hangs, no FLAT faults, no Triton kernels |

**TrainingArguments**  
```
num_train_epochs=3
per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=2e-4
bf16=True
logging_steps=10
save_steps=500
optim="adamw_torch"
gradient_checkpointing=True
remove_unused_columns=False
max_seq_length=2048
```

**Hardware Telemetry**  
Pre‑run
- Temp: 38°C
- Power: 22W
- VRAM: 6%

Post‑run
- Temp: 64°C
- Power: 121W
- VRAM: 82%

System RAM
- 3.1 GiB → 5.2 GiB


---

#  Appendix A — Production Models
A list of currently deployed models, their versions, and their merge paths.  


