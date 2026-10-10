"""Verify the fixed, separately distributed simulation example assets."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path


def verify_simulation_assets(root: str | Path) -> dict:
    manifest = json.loads(files("clothdojo.assets").joinpath("simulation_manifest.json").read_text())
    root = Path(root).expanduser()
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing simulation asset: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"simulation asset checksum differs: {path}")
    return manifest
