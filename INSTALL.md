# Installation Guide (ROCm + PyTorch + QLoRA) — Ubuntu 24.04 (Noble)

This guide walks you through setting up a full AMD ROCm environment with PyTorch (ROCm build), a Python 3.10 virtual environment, and all dependencies required for QLoRA training on RDNA3 GPUs such as the RX 7700 XT / 7800 XT.

---

## 1. System Requirements
- Ubuntu 24.04 LTS (Noble)
- AMD RDNA3 GPU (e.g., RX 7700 XT, 7800 XT)
- Kernel 6.x (default on 24.04)
- Python 3.10 (required for PyTorch ROCm wheels)
- At least 16GB system RAM
- 12GB+ VRAM recommended for 3B–7B models

---

## 1.1 Install Python 3.10 (required for ROCm PyTorch)

Ubuntu 24.04 ships with Python 3.12, which is not supported by PyTorch ROCm wheels.
Install Python 3.10 from deadsnakes:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

Verify:

```bash
python3.10 --version
```

Create your virtual environment using Python 3.10:

```bash
python3.10 -m venv ~/rocm-env
```

---

## 2. Install ROCm (Ubuntu 24.04)

### Add AMD ROCm repository for Noble
```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.1 noble main' | sudo tee /etc/apt/sources.list.d/rocm.list
```

### Install ROCm packages
```bash
sudo apt update
sudo apt install rocm-hip-sdk rocm-hip-libraries rocm-device-libs
```

### Add ROCm to environment
```bash
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Add user to required groups
```bash
sudo usermod -aG video $USER
sudo usermod -aG render $USER
```

### Add ROCm to linker path
```bash
echo /opt/rocm/lib | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig
```

### Reboot
```bash
sudo reboot
```

---

## 3. Verify ROCm Installation

### Check GPU visibility
rocminfo | grep -i gfx

Expected: gfx1101 (RDNA3)

### Verify HIP runtime
hipinfo

---

## 4. Create Python Virtual Environment
```bash
python3.10 -m venv ~/rocm-env
source ~/rocm-env/bin/activate
pip install --upgrade pip
```

---

## 5. Install PyTorch (ROCm Build)

### Install from ROCm wheel index
```bash
pip install torch torchvision torchaudio \
  --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/
```

### Verify PyTorch sees your GPU
```bash
python3 - << 'EOF'
import torch
print("HIP version:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
EOF
```

Expected output:
- HIP version: 6.1
- GPU available: True
- Device: AMD Radeon RX 7700 XT (or similar)

---

## 6. Install QLoRA + Training Dependencies
```bash
pip install transformers accelerate datasets sentencepiece
pip install peft
```

### Install Quanto (ROCm‑safe 4‑bit quantization)
```bash
pip install "quanto>=0.2.0"
```

### Install Unsloth (Optional)
Unsloth can accelerate QLoRA training, but must be installed **without CUDA extras** to avoid overwriting ROCm PyTorch.

```bash
pip install unsloth
# or
pip install git+https://github.com/unslothai/unsloth.git
```

---

## 7. RDNA3 / ROCm Stability Fixes (Credits: BEATEK_ROCm)

These environment variables significantly improve stability for RDNA3 GPUs during QLoRA training and inference.  
Originally documented by Beat‑k in the BEATEK_ROCm project:

https://github.com/Beat-k/BEATEK_ROCm

Add the following to your `~/.bashrc`:

```bash
# RDNA3 / ROCm stability fixes for QLoRA + inference
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export HSA_OVERRIDE_GFX_VERSION="11.0.0"
export HSA_ENABLE_SDMA=1
export ROCM_FORCE_ENABLE_DP=1
```

Reload your shell:

```bash
source ~/.bashrc
```

---

## 8. Download Model Weights (Recommended to log in to HF beforehand)

```bash
pip install huggingface_hub
huggingface-cli login
```

Example:

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir ./models/qwen25-3b
```

---

## 9. Run a Test Script
```bash
python3 - << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("./models/qwen25-3b")
model = AutoModelForCausalLM.from_pretrained("./models/qwen25-3b", device_map="auto")

print("Loaded successfully.")
EOF
```

If this runs without errors, your environment is ready.

---

## 10. RDNA3 Notes
- ROCm 6.1 includes major RDNA3 improvements.
- This pipeline uses Quanto 4‑bit quantization, which is fully compatible with ROCm and RDNA3. BitsAndBytes is not used because its CUDA/Triton kernels are incompatible with AMD GPUs.
- The RDNA3 inference fix significantly improves stability.

---

## 11. Troubleshooting

### PyTorch cannot find GPU
- Ensure `rocminfo` shows `gfx1101`
- Ensure `torch.version.hip` prints a version
- Ensure you rebooted after adding user to groups

### If training stalls, check:
  - hipBLASLt warnings
  - mismatched PyTorch wheels
  - missing ROCm libraries

---

### Your ROCm + PyTorch environment is now fully configured for QLoRA training.

