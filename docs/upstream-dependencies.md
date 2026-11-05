# Upstream dependencies

[Trajectory generation](usage.md) · [Downloads and assets](data-and-assets.md)

Run these commands from the GitHub checkout root. Obtain upstream files under their publisher's terms; this guide does not redistribute them.

## Tokenizer for NOCS inference

The NOCS teachers contain their trained weights and normalization, but not the PaliGemma tokenizer. The pinned OpenPI installation obtains it from `gs://big_vision/paligemma_tokenizer.model`, as shown in the [upstream tokenizer implementation](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py).

The trajectory generation guide includes cache preparation in its policy-server commands. For another policy-server or benchmark terminal, use the same cache:

```bash
export CLOTHDOJO_OPENPI_CACHE="$PWD/.cache/openpi"
export OPENPI_DATA_HOME="$CLOTHDOJO_OPENPI_CACHE"
.venv/bin/python - <<'PY'
from openpi.shared import download
path = download.maybe_download(
    "gs://big_vision/paligemma_tokenizer.model",
    gs={"token": "anon", "session_kwargs": {"trust_env": True}},
)
print(path)
PY
```

The resulting file must be at `.cache/openpi/big_vision/paligemma_tokenizer.model`. Set both variables in every terminal that starts a policy server or managed benchmark. They must point to the same cache.

If the upstream download is denied or inaccessible, obtain this tokenizer through the publisher's authorized access route and place it at that exact path. An arbitrary tokenizer or a different Gemma tokenizer is not interchangeable. The teacher repository's `base_model.json`, notices, and Gemma restrictions identify the upstream model and applicable terms.

## Isaac 5 runtime

The [randomized RGB guide](randomized-rendering.md#1-prepare-the-renderer) contains the complete pinned Isaac 5.0 installation command, public-material download and first export. Complete that path before adding the optional resources below.

## NVIDIA visual assets for RGB

The full appearance profiles use original scenes, MDL materials and textures omitted from the asset package. These are additional resources beyond the basic Poly Haven example; installing the SDK does not restore them.

1. Obtain Isaac Sim **5.0** visual assets from NVIDIA under its applicable terms. Use the [official download entry](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) and select the matching archived version. The [5.0 asset browser](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/utilities/asset_browser.html) describes upstream browsing and downloading. A newer asset version is not an exact replacement for this snapshot.
2. Locate the downloaded asset root containing both `Isaac/` and `NVIDIA/`. Copy the matching original files under `data/visual-assets/isaac-5.0/`, preserving the relative paths in the omission manifest. Keep these files inside the visual-assets tree: the extraction code records dependencies relative to that tree, so an external directory symlink is unsuitable.
3. Reconstruct the selected extracted materials from those local originals with the existing runtime function:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from clothdojo.assets.scene_materials import extract_scene_materials
extract_scene_materials(Path("data/visual-assets"))
PY
```

The function checks the source material hashes and recreates the MDL wrappers, texture copies, and extraction manifests. It also processes the included GenieSim material sources.

4. Verify every omitted dependency against the release manifest:

```bash
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

missing, changed = [], []
for item in json.loads(Path("upstream_visual_dependencies.json").read_text()):
    path = Path(item["path"])
    if not path.is_file():
        missing.append(str(path))
        continue
    if path.stat().st_size != item["bytes"]:
        changed.append(str(path))
        continue
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["sha256"]:
        changed.append(str(path))
print(f"Missing: {len(missing)}; different: {len(changed)}")
for path in missing + changed:
    print(path)
raise SystemExit(bool(missing or changed))
PY
```

Proceed with the exact released appearance setup only when this check reports zero missing and zero different files. If it fails, use the listed paths and hashes to identify the version or extraction mismatch. Missing upstream files are not repaired by renaming a newer asset or substituting a different background.

Use the omission manifest as the file, size and hash reference for restoring these full appearance resources.

## What to download first

For the shortest NOCS path, follow the [Quickstart](usage.md). Then try [randomized RGB rendering](randomized-rendering.md) with the public materials. Restore NVIDIA visual assets when using the full appearance profiles. The RGB training datasets are not required to generate or render new trajectories.
