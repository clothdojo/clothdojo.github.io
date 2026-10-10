"""Stage the exact legacy evaluation inputs as a portable benchmark candidate.

This is a maintainer tool. It requires access to the private legacy project;
the staged output is not automatically licensed for redistribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


CONDITIONS = {
    "nocs-flatten": "dagger_hil800_aggregate_strict_40k_s43_chunk_h10_balanced200_v2_large_support_60s_rescore_20260908",
    "nocs-fold": "fold_nocs_prefix_verified40k_eval800_20260912",
    "rgb-flatten": "pi05_rgb_flatten_generated_unified100x100_dualtex_v1_instantaneous_score",
    "rgb-fold": "pi05_rgb_fold_generated_unified100x100_dualtex_v2",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def state_index(legacy: Path) -> dict[tuple[str, str], tuple[Path, int | None]]:
    roots = [legacy / "data/benchmarks/balanced200_v2_20260907/states"]
    roots += sorted(p for p in (legacy / "data/states").iterdir() if (p / "meta.json").exists())
    roots += [legacy / "outputs/evaluations/fold_noaug_eval800_20260909/pools"]
    index = {}
    for root in roots:
        for item in read_json(root / "meta.json").get("assets", []):
            if isinstance(item, dict) and item.get("asset_id") and item.get("sha256"):
                index[(item["asset_id"], item["sha256"])] = (
                    root / item["path"], item.get("state_count")
                )
    return index


def template_index(legacy: Path) -> dict[tuple[str, str], Path]:
    index = {}
    for meta_path in sorted((legacy / "data/benchmarks").glob("*/meta.json")):
        for item in read_json(meta_path).get("assets", []):
            if isinstance(item, dict) and item.get("asset_id") and item.get("sha256") and item.get("path"):
                index[(item["asset_id"], item["sha256"])] = meta_path.parent / item["path"]
    # These frozen template collections predate the per-template meta schema.
    for name in ("balanced200_20260901", "large_support_v1_20260908"):
        for path in (legacy / "data/benchmarks" / name / "templates/templates").glob("*.npz"):
            index[(path.stem, digest(path))] = path
    return index


def checked_copy(source: Path, target: Path, output: Path, expected: str | None = None) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = digest(source)
    if expected and actual != expected:
        raise ValueError(f"SHA-256 mismatch: {source}: {actual} != {expected}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or digest(target) != actual:
        shutil.copy2(source, target)
    if digest(target) != actual:
        raise ValueError(f"Copied file failed verification: {target}")
    return {"path": str(target.relative_to(output)), "sha256": actual, "bytes": source.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    legacy = args.legacy_root.resolve()
    output = args.output.resolve()
    states = state_index(legacy)
    templates = template_index(legacy)
    manifest = {"schema": "clothdojo_benchmark_release_candidate", "version": 1, "conditions": {}, "garments": {}}

    for label, evaluation_name in CONDITIONS.items():
        evaluation_path = legacy / "outputs/evaluations" / evaluation_name / "meta.json"
        evaluation = read_json(evaluation_path)
        if label == "nocs-fold":
            pool_meta = read_json(legacy / "outputs/evaluations/fold_noaug_eval800_20260909/pools/meta.json")
            expected_states = {item["asset_id"]: item["sha256"] for item in pool_meta["assets"]}
            expected_templates = {item["asset_id"]: item["flat_reference_sha256"] for item in pool_meta["assets"]}
            fold_template_paths = {item["asset_id"]: Path(item["flat_reference"]) for item in pool_meta["assets"]}
            asset_splits = {item["asset_id"]: item["split"] for item in pool_meta["assets"]}
            if len(evaluation["trajectories"]) != 800 or {item["asset_id"] for item in evaluation["trajectories"]} != set(expected_states):
                raise ValueError("Folding evaluation and frozen pool asset IDs differ")
        else:
            expected_states = evaluation["initial_state_pools"]
            expected_templates = evaluation["templates"]
            asset_splits = evaluation["benchmark"]["asset_splits"]
        if len(expected_states) != 200 or set(expected_states) != set(expected_templates):
            raise ValueError(f"Unexpected panel size or IDs for {label}")

        condition = {
            "evaluation_meta_sha256": digest(evaluation_path),
            "source_evaluation": evaluation_name,
            "cloth_visual": "rgb" if label.startswith("rgb") else "nocs",
            "asset_count": 200,
            "evaluation_cases_per_asset": 8 if label.startswith("rgb") else 4,
            "case_count": len(evaluation["trajectories"]),
            "protocol": evaluation.get("scoring_protocol", evaluation.get("protocol")),
            "scenario": evaluation.get("scenario"),
            "observation": evaluation.get("observation"),
            "assets": {},
        }
        protocol = evaluation.get("scoring_protocol")
        if label == "nocs-fold":
            # The original rollout meta predates embedded scoring definitions.
            # The paper's fold summary uses the same 40 s / 3 s-hold Q contract
            # recorded in the later RGB fold evaluation.
            donor = legacy / "outputs/evaluations" / CONDITIONS["rgb-fold"] / "meta.json"
            protocol = read_json(donor)["scoring_protocol"]
            condition["protocol"] = protocol
            condition["protocol_source_evaluation"] = CONDITIONS["rgb-fold"]
        if protocol is not None:
            protocol_path = output / "conditions" / label / "protocol.json"
            protocol_path.parent.mkdir(parents=True, exist_ok=True)
            protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
            condition["protocol_file"] = {
                "path": str(protocol_path.relative_to(output)),
                "sha256": digest(protocol_path),
                "bytes": protocol_path.stat().st_size,
            }
        else:
            raise ValueError(f"No scoring protocol for {label}")
        for asset_id in sorted(expected_states):
            state_match = states.get((asset_id, expected_states[asset_id]))
            template_source = templates.get((asset_id, expected_templates[asset_id]))
            if template_source is None and label == "nocs-fold":
                template_source = fold_template_paths[asset_id]
            if state_match is None or template_source is None:
                raise FileNotFoundError(f"Missing matching state/template for {label}/{asset_id}")
            state_source, state_count = state_match
            state_target = output / "conditions" / label / "states" / asset_id / "states.npz"
            template_target = output / "conditions" / label / "templates" / f"{asset_id}.npz"
            condition["assets"][asset_id] = {
                "state": checked_copy(state_source, state_target, output, expected_states[asset_id]),
                "template": checked_copy(template_source, template_target, output, expected_templates[asset_id]),
                "physical_state_count": state_count,
                "split": asset_splits[asset_id],
            }
            garment_source = legacy / "assets/garments/converted" / asset_id / f"{asset_id}.usdc"
            if asset_id not in manifest["garments"]:
                garment_target = output / "assets/garments/converted" / asset_id / f"{asset_id}.usdc"
                manifest["garments"][asset_id] = checked_copy(garment_source, garment_target, output)
        manifest["conditions"][label] = condition
        print(f"{label}: verified {len(condition['assets'])} garments and state/template pairs", flush=True)

    manifest["robot_assets"] = {}
    for source in sorted((legacy / "assets/robots/piper").rglob("*")):
        if source.is_file():
            relative = source.relative_to(legacy / "assets")
            manifest["robot_assets"][str(relative)] = checked_copy(
                source, output / "assets" / relative, output
            )
    example_garment = legacy / "assets/garments/converted/rand_ZZCWITH5HY/rand_ZZCWITH5HY.usdc"
    manifest["simulation_example_garment"] = checked_copy(
        example_garment,
        output / "assets/garments/converted/rand_ZZCWITH5HY/rand_ZZCWITH5HY.usdc",
        output,
    )

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output / 'manifest.json'} with {len(manifest['garments'])} unique garments")


if __name__ == "__main__":
    main()
