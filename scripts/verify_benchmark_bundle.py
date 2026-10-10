"""Verify every file in a staged ClothDojo evaluation bundle."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--semantic", action="store_true", help="also load every pool and template")
    args = parser.parse_args()
    root = args.bundle.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "clothdojo_benchmark_release_candidate":
        parser.error("unsupported benchmark manifest")
    files = (list(manifest["garments"].values())
             + list(manifest["robot_assets"].values())
             + [manifest["simulation_example_garment"]])
    for name, condition in manifest["conditions"].items():
        if len(condition["assets"]) != condition["asset_count"]:
            raise ValueError(f"{name}: asset count mismatch")
        splits = Counter(item["split"] for item in condition["assets"].values())
        print(f"{name}: {len(condition['assets'])} assets, splits {dict(splits)}", flush=True)
        files.append(condition["protocol_file"])
        for item in condition["assets"].values():
            files.extend((item["state"], item["template"]))
    unique_files = {item["path"]: item for item in files}
    for item in unique_files.values():
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Missing or corrupt benchmark input: {path}")
    print(f"Verified {len(unique_files)} payload files in {root}")
    if args.semantic:
        from clothdojo.assets.bundle import verify_simulation_assets
        from clothdojo.evaluation.metric import load_canonical_template
        from clothdojo.rollout.initial_states import InitialStatePool, validate_pool_compatibility

        verify_simulation_assets(root / "assets")
        for name, condition in manifest["conditions"].items():
            counts = Counter()
            for asset_id, item in condition["assets"].items():
                pool = InitialStatePool.load(root / item["state"]["path"])
                template = load_canonical_template(root / item["template"]["path"])
                garment = root / manifest["garments"][asset_id]["path"]
                validate_pool_compatibility(
                    pool, garment_path=garment, particle_count=template.particle_count
                )
                if (template.asset_id != pool.metadata["garment_id"]
                        or template.asset_sha256 != pool.metadata["garment_sha256"]):
                    raise ValueError(f"{name}/{asset_id}: template and pool differ")
                if item["physical_state_count"] != pool.state_count:
                    raise ValueError(f"{name}/{asset_id}: state count differs")
                counts[pool.state_count] += 1
            print(f"{name}: loaded {len(condition['assets'])} pool/template pairs; state counts {dict(counts)}")


if __name__ == "__main__":
    main()
