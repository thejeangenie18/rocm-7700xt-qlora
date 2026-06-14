#!/usr/bin/env python3
"""
validate_all.py

General-purpose, open-source-safe validation script for HuggingFace models
and optional PEFT LoRA adapters. Designed for AMD RDNA3 / ROCm environments.

This script performs:
- Base model sanity checks
- Optional LoRA adapter checks
- Forward-pass validation
- Dtype consistency
- Parameter shape compatibility
- KL divergence (base vs adapter)
- Entropy shift analysis
- Token distribution shift
- Merge-safety diagnostics (no merge performed)
- JSON reporting

No CUDA-only kernels, no FlashAttention, no Triton, no project-specific imports.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from peft import PeftModel, PeftConfig


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def now_local():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")  # ROCm maps here
    return torch.device("cpu")


def load_model_and_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    last_err = None
    for loader in (AutoModelForCausalLM, AutoModelForSeq2SeqLM):
        try:
            model = loader.from_pretrained(
                path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            return model, tok
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to load model from {path}: {last_err}")


def load_adapter(model, adapter_path: str):
    _ = PeftConfig.from_pretrained(adapter_path)
    return PeftModel.from_pretrained(model, adapter_path, device_map="auto")


def dummy_inputs(tok, device, max_len=64):
    text = "Validation test input."
    enc = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return {k: v.to(device) for k, v in enc.items()}


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

def test_forward(model, inputs):
    out = {"name": "forward_pass", "passed": False, "warning": None, "error": None}
    try:
        with torch.no_grad():
            _ = model(**inputs)
        out["passed"] = True
    except Exception as e:
        out["error"] = str(e)
    return out


def test_lora_presence(adapter):
    out = {"name": "lora_presence", "passed": False, "warning": None, "error": None}
    names = [n for n, _ in adapter.named_parameters() if "lora_" in n.lower()]
    if names:
        out["passed"] = True
    else:
        out["warning"] = "No LoRA parameters detected."
    return out


def test_dtype(base, adapter):
    out = {"name": "dtype_consistency", "passed": True, "warning": None, "error": None}
    bd = {p.dtype for p in base.parameters()}
    ad = {p.dtype for p in adapter.parameters()}
    if bd != ad:
        out["warning"] = f"Base dtypes {bd} differ from adapter dtypes {ad}"
    return out


def test_shape(base, adapter):
    out = {"name": "shape_compatibility", "passed": True, "warning": None, "error": None}
    mism = []
    bs = base.state_dict()
    as_ = adapter.state_dict()
    for k, v in as_.items():
        if k in bs and bs[k].shape != v.shape:
            mism.append(f"{k}: base {bs[k].shape} vs adapter {v.shape}")
    if mism:
        out["passed"] = False
        out["error"] = "\n".join(mism)
    return out


def softmax(x):
    return torch.softmax(x, dim=-1)


def kl(p, q):
    eps = 1e-8
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def test_kl(base_logits, adapter_logits):
    out = {"name": "kl_divergence", "passed": True, "warning": None, "error": None}
    p = softmax(base_logits)
    q = softmax(adapter_logits)
    k = kl(p, q).mean().item()
    out["kl_mean"] = k
    if k < 1e-5:
        out["warning"] = "KL extremely low; adapter may not change behavior."
    if k > 2.0:
        out["warning"] = "KL high; adapter may destabilize behavior."
    return out


def entropy(p):
    eps = 1e-8
    p = p.clamp(min=eps)
    return -(p * p.log()).sum(dim=-1)


def test_entropy_shift(base_logits, adapter_logits):
    out = {"name": "entropy_shift", "passed": True, "warning": None, "error": None}
    p = softmax(base_logits)
    q = softmax(adapter_logits)
    e1 = entropy(p).mean().item()
    e2 = entropy(q).mean().item()
    out["base_entropy"] = e1
    out["adapter_entropy"] = e2
    if abs(e1 - e2) > 1.0:
        out["warning"] = "Large entropy shift; adapter may be over/under confident."
    return out


def test_token_shift(base_logits, adapter_logits):
    out = {"name": "token_distribution_shift", "passed": True, "warning": None, "error": None}
    top_k = 10
    b = base_logits.topk(top_k, dim=-1).indices[0, -1]
    a = adapter_logits.topk(top_k, dim=-1).indices[0, -1]
    overlap = len(set(b.tolist()) & set(a.tolist())) / top_k
    out["overlap"] = overlap
    if overlap < 0.2:
        out["warning"] = "Low top-k overlap; adapter significantly shifts token preferences."
    return out


def test_merge_safety(base, adapter):
    out = {"name": "merge_safety_diagnostic", "passed": True, "warning": None, "error": None}
    bs = base.state_dict()
    as_ = adapter.state_dict()
    bad = []
    for k, v in as_.items():
        if k in bs and bs[k].shape != v.shape:
            bad.append(k)
    if bad:
        out["passed"] = False
        out["error"] = "Incompatible shapes: " + ", ".join(bad)
    return out


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def run(base_path, adapter_path=None):
    device = choose_device()
    start = time.time()

    report = {
        "iso_timestamp": now_iso(),
        "local_timestamp": now_local(),
        "base_model": base_path,
        "adapter": adapter_path,
        "device": str(device),
        "tests": [],
        "warnings": [],
        "errors": [],
        "execution_seconds": None,
    }

    try:
        base, tok = load_model_and_tokenizer(base_path)
        base.to(device)
    except Exception as e:
        report["errors"].append(f"Failed to load base model: {e}")
        return report

    adapter = None
    if adapter_path:
        try:
            adapter = load_adapter(base, adapter_path)
            adapter.to(device)
        except Exception as e:
            report["errors"].append(f"Failed to load adapter: {e}")
            return report

    # Dummy inputs
    inputs = dummy_inputs(tok, device)

    # Forward passes
    try:
        with torch.no_grad():
            base_logits = base(**inputs).logits
            adapter_logits = adapter(**inputs).logits if adapter else None
    except Exception as e:
        report["errors"].append(f"Forward pass failed: {e}")
        return report

    # Run tests
    def add(t):
        report["tests"].append(t)
        if t.get("warning"):
            report["warnings"].append(f"{t['name']}: {t['warning']}")
        if t.get("error"):
            report["errors"].append(f"{t['name']}: {t['error']}")

    add(test_forward(base, inputs))
    if adapter:
        add(test_forward(adapter, inputs))
        add(test_lora_presence(adapter))
        add(test_dtype(base, adapter))
        add(test_shape(base, adapter))
        add(test_merge_safety(base, adapter))
        add(test_kl(base_logits, adapter_logits))
        add(test_entropy_shift(base_logits, adapter_logits))
        add(test_token_shift(base_logits, adapter_logits))

    report["execution_seconds"] = time.time() - start
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--failed-only", action="store_true")
    args = ap.parse_args()

    rep = run(args.base_model, args.adapter)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(rep, f, indent=2)

    if args.failed_only:
        minimal = {
            "errors": rep["errors"],
            "warnings": rep["warnings"],
            "failed_tests": [t for t in rep["tests"] if not t.get("passed", True)],
        }
        print(json.dumps(minimal, indent=2))
    else:
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
