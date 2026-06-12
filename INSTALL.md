# INSTALL.md — ROCm 7.2.1 + PyTorch + QLoRA (RDNA3 / Ubuntu 24.04)
A complete guide for setting up a stable, RDNA3‑safe QLoRA training environment using ROCm 7.2.1, PyTorch ROCm wheels, and Python 3.10.
Validated on Radeon RX 7700 XT with ROCm 7.2.1 across multiple real training runs (TinyLlama, Qwen3B incremental).

This environment uses:  
- No Triton
- No bitsandbytes
- No CUDA‑only kernels
- BF16‑only compute
- ROCm‑native kernels only

---

## 1. System Requirements
- Ubuntu 24.04 LTS (Noble)
- AMD **RDNA3 GPU** (RX 7700 XT, 7800 XT, 7900 GRE, etc.)
- ROCm 7.2.1
- Python 3.10 (required for PyTorch ROCm wheels)
- 16GB+ RAM
- 12GB+ VRAM recommended for 3B–7B models

---

## 1.1 Install Python 3.10 (required for ROCm PyTorch)
Ubuntu 24.04 ships with Python 3.12, which is not supported by PyTorch ROCm wheels.
Install Python 3.10:  
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

## 2. Install ROCm 7.2.1 (Ubuntu 24.04)

### Add AMD ROCm repository for Noble
```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
  | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/7.2 noble main' \
  | sudo tee /etc/apt/sources.list.d/rocm.list
```

### Install ROCm 7.2.1 packages
```bash
sudo apt update
sudo apt install rocm-hip-sdk rocm-hip-runtime rocm-device-libs rocm-opencl-runtime rocminfo
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
```bash
rocminfo | grep -i gfx
```

Expected: ```gfx1101``` (RDNA3)

## Check ROCm version
```bash
dpkg -l | grep rocm
```
Expected: ```7.2.1.70201-81~24.04```

### Verify HIP runtime
```bash
hipinfo
```

---

## 4. Create Python Virtual Environment
```bash
python3.10 -m venv ~/rocm-env
source ~/rocm-env/bin/activate
pip install --upgrade pip
```

---

## 5. Install PyTorch (ROCm 7.2 Build)

### Install from the ROCm 7.2 wheel index:
```bash
pip install torch torchvision \
  --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
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
- HIP version: 7.2
- GPU available: True
- Device: AMD Radeon RX 7700 XT (or similar)

---

## 6. Install QLoRA + Training Dependencies
```bash
pip install \
  accelerate==1.13.0 \
  datasets==4.8.5 \
  einops==0.8.2 \
  huggingface_hub==1.17.0 \
  loguru==0.7.3 \
  numpy==2.2.6 \
  orjson==3.11.9 \
  packaging==26.2 \
  peft==0.19.1 \
  pillow==12.2.0 \
  python-dotenv==1.0.1 \
  regex==2026.5.9 \
  requests==2.32.3 \
  rich==13.7.1 \
  safetensors==0.7.0 \
  sentencepiece==0.2.1 \
  tqdm==4.67.3 \
  transformers==5.9.0 \
  typing_extensions==4.15.0 \
  urllib3==2.7.0
```

### Important:
- No Triton
- No bitsandbytes
- No ONNX Runtime
- No CUDA‑only packages
- **BF16 compute only** (FP16 is unsafe on RDNA3)

# Optional: Install Quanto (ROCm‑safe 4‑bit quantization)
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
Install New CLI:
```bash
pip install huggingface_hub
```

Login:
```
hf auth login
```

Verify Login:
```bash
hf auth whoami
```

Example: 
```bash
hf download Qwen/Qwen2.5-3B-Instruct \
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
EOF
```

If this runs without errors, your environment is ready.

---

## 10. RDNA3 Notes (ROCm 7.2.1)
**Confirmed safe:**
- BF16 compute
- ROCm‑native kernels
- hipBLASLt → hipBLAS fallback
- SDMA enabled
- No Triton
- No FlashAttention
- No bitsandbytes

**Confirmed unsafe:**  
- FP16 matmuls
- Triton kernels
- hipBLASLt (without fallback)
- Partial‑wait FLAT/SMEM sequences
- Predicated WMMA tiles
---

## 11. Troubleshooting

### PyTorch cannot find GPU
- Ensure `rocminfo` shows `gfx1101`
- Ensure `torch.version.hip` prints `7.2`
- Ensure you rebooted after adding user to groups

### If training stalls, check:
- Check for hipBLASLt warnings
- Ensure PyTorch was installed from the ROCm 7.2 wheel index
- Ensure Triton is not installed
- Ensure BF16 is being used

### Merge Failures
- Ensure adapter path is correct
- Ensure model + adapter use matching architectures
- Ensure you are not mixing FP16 + BF16 weights
---

### Your ROCm 7.2.1 + PyTorch environment is now fully configured for QLoRA training on RDNA3.

