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
release_dir = RELEASES / f"adapter_v1.0_{timestamp}"
release_dir.mkdir()

# -----------------------------
# Files to freeze
# -----------------------------
files = [
    "adapter_model.safetensors",
    "adapter_config.json"
]

# -----------------------------
# Validate source files exist
# -----------------------------
missing = [f for f in files if not (SOURCE / f).exists()]
if missing:
    raise FileNotFoundError(
        f"❌ Missing adapter files: {missing}\n"
        f"Expected in: {SOURCE}"
    )

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
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# -----------------------------
# Write manifest
# -----------------------------
manifest = release_dir / "MANIFEST.json"

manifest_data = {
    "version": "v1.0",
    "timestamp": timestamp,
    "source_dir": str(SOURCE),
    "release_dir": str(release_dir),
    "files": {
        fname: {
            "sha256": sha256(release_dir / fname),
            "size_bytes": (release_dir / fname).stat().st_size
        }
        for fname in files
    }
}

manifest.write_text(json.dumps(manifest_data, indent=4))

print(f"[FROZEN] Adapter frozen at: {release_dir}")
