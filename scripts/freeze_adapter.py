#!/usr/bin/env python3
import shutil
import hashlib
import datetime
from pathlib import Path
import json

# -----------------------------
# Paths
# -----------------------------
SOURCE = Path("./models/adapter")
RELEASES = Path("./releases")
RELEASES.mkdir(exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
release_dir = RELEASES / f"adapter_v0.1_demo_{timestamp}"
release_dir.mkdir()

# -----------------------------
# Files to freeze
# -----------------------------
files = [
    "adapter_model.safetensors",
    "adapter_config.json"
]

# -----------------------------
# Copy files
# -----------------------------
for fname in files:
    src = SOURCE / fname
    dst = release_dir / fname
    shutil.copy2(src, dst)

# -----------------------------
# Hash function
# -----------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# -----------------------------
# Write manifest
# -----------------------------
manifest = release_dir / "MANIFEST.json"
manifest.write_text(
    json.dumps(
        {
            "version": "v0.1-demo",
            "timestamp": timestamp,
            "files": {
                fname: sha256(release_dir / fname)
                for fname in files
            }
        },
        indent=4
    )
)

print(f"[FROZEN] Adapter frozen at: {release_dir}")
