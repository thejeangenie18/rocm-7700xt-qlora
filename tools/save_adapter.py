#!/usr/bin/env python3
"""
save_adapter.py
Save adapter directory with metadata and register it in adapters/registry.json
Usage:
  python tools/save_adapter.py --src ./output/checkpoint-lora --meta metrics.json
"""

import argparse
import json
import os
import shutil
from datetime import datetime
import hashlib

REGISTRY = "adapters/registry.json"

def hash_dir(path):
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            with open(fpath, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--meta", required=False)
    p.add_argument("--outdir", default="adapters", help="Adapters root")
    args = p.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest = os.path.join(args.outdir, f"adapter-{ts}")
    os.makedirs(dest, exist_ok=True)
    shutil.copytree(args.src, os.path.join(dest, "files"))
    meta = {}
    if args.meta and os.path.exists(args.meta):
        meta = json.load(open(args.meta, "r", encoding="utf-8"))
    meta["saved_at"] = ts
    meta["src"] = args.src
    meta["dir_hash"] = hash_dir(os.path.join(dest, "files"))
    with open(os.path.join(dest, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # update registry
    os.makedirs(args.outdir, exist_ok=True)
    registry = []
    if os.path.exists(REGISTRY):
        registry = json.load(open(REGISTRY, "r", encoding="utf-8"))
    registry.append({"name": os.path.basename(dest), "path": dest, "metadata": meta})
    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print("Adapter saved to", dest)
    print("Registry updated at", REGISTRY)

if __name__ == "__main__":
    main()
