"""Audit state geometry and repair slots identified by confirmed human review."""

from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.assets import GarmentLibrary
from clothdojo.rollout.initial_states import (
    InitialStatePool,
    _atomic_write_json,
    _run_state_workers,
    _sha256,
    load_initial_state_collection,
)
from clothdojo.rollout.state_geometry import (
    geometry_diagnostics_settings,
    measure_self_intersections,
)
from clothdojo.rollout.state_review import inspect_sequential_hil_review
from clothdojo.sim.cloth import load_garment_mesh


def geometry_validation_settings() -> dict[str, Any]:
    return geometry_diagnostics_settings()


def inspect_pool_geometry(pool: InitialStatePool, garment_path: str | Path) -> list[dict]:
    """Recompute from the saved particles; old validation metadata is not evidence."""

    if _sha256(Path(garment_path)) != pool.metadata.get("garment_sha256"):
        raise ValueError("initial-state pool belongs to a different garment asset")
    rest, indices, _, _ = load_garment_mesh(garment_path)
    from clothdojo.sim.scene import ShirtFlattenSpec

    # Older pools predate explicit scale metadata and used the fixed scene scale.
    scale = float(pool.metadata.get("garment_scale", ShirtFlattenSpec().garment_scale))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid garment scale in initial-state pool")
    geometry = measure_self_intersections(pool.particle_q, indices.reshape(-1, 3), rest * scale)
    return [
        {
            "index": index,
            "seed": int(pool.seeds[index]),
            "geometry_flagged": bool(geometry.severe[index]),
            "self_intersection_score": float(geometry.severity_score[index]),
            "self_intersection_count": int(count),
            "self_intersection_fraction": float(fraction),
            "self_intersection_length_m": float(geometry.total_length_m[index]),
            "self_intersection_maximum_component_m": float(
                geometry.maximum_component_length_m[index]
            ),
            "self_intersection_region_fraction": float(geometry.maximum_region_fraction[index]),
            "self_intersection_region_radius_m": float(geometry.maximum_region_radius_m[index]),
        }
        for index, (count, fraction) in enumerate(
            zip(geometry.count, geometry.rest_area_fraction, strict=True)
        )
    ]


def check_initial_state_collection(
    source: str | Path,
    *,
    assets_root: str | Path = "assets",
    report_path: str | Path,
    repair_output: str | Path | None = None,
    gpus: tuple[str, ...] = ("0",),
    review_rollout: str | Path | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Geometry is diagnostic; replacements require an explicit initial-defect cause."""

    root = Path(source).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    assets = Path(assets_root).expanduser().resolve()
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    meta = load_initial_state_collection(root, allow_rejected=True)
    source_sha256 = _sha256(root / "meta.json")
    review = None if review_rollout is None else inspect_sequential_hil_review(review_rollout)
    if repair_output is not None and review is None:
        raise ValueError("repair requires --review-rollout and confirmed sequential skip semantics")
    reviewed_states = (
        {}
        if review is None
        else {
            (entry["pool_sha256"], entry["initial_state_index"]): entry["decision"]
            for entry in review["entries"]
        }
    )
    rejected_assets = set() if review is None else set(review["rejected_assets"])
    jobs = [(root, assets, item) for item in meta["assets"]]
    if workers == 1:
        inspected = list(map(_inspect_collection_asset, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            inspected = list(pool.map(_inspect_collection_asset, jobs))
    records = []
    for item, (pool_sha256, states) in zip(meta["assets"], inspected, strict=True):
        asset_id = item["asset_id"]
        for state in states:
            state["review_decision"] = (
                "rejected_asset"
                if asset_id in rejected_assets
                else reviewed_states.get((pool_sha256, state["index"]), "unreviewed")
            )
            state["replace"] = state["review_decision"] == "initial_self_intersection"
        records.append(
            {
                "asset_id": asset_id,
                "sha256": pool_sha256,
                "states": states,
                "exclude_asset": asset_id in rejected_assets,
            }
        )
    invalid_count = sum(state["replace"] for item in records for state in item["states"])
    report = {
        "schema": "clothdojo_initial_state_geometry_audit",
        "source": str(root),
        "source_meta_sha256": source_sha256,
        "geometry_validation": geometry_validation_settings(),
        "state_count": sum(len(item["states"]) for item in records),
        "invalid_state_count": invalid_count,
        "excluded_asset_count": sum(r["exclude_asset"] for r in records),
        "geometry_flagged_count": sum(s["geometry_flagged"] for r in records for s in r["states"]),
        "sequential_review": review,
        "assets": records,
    }
    if review is not None:
        states = [s for r in records for s in r["states"]]
        positives = [s for s in states if s["review_decision"] == "initial_self_intersection"]
        marked = [s for s in states if s["review_decision"] == "marked"]
        report["hil_validation"] = {
            "rule": geometry_diagnostics_settings()["method"],
            "adopted_for_generation": True,
            "status": "diagnostic" if positives else "no_confirmed_initial_defect_examples",
            "penetration_examples": len(positives),
            "penetration_detected": sum(s["geometry_flagged"] for s in positives),
            "penetration_missed": sum(
                not s["geometry_flagged"] for s in positives
            ),
            "marked_examples": len(marked),
            "marked_flagged": sum(s["geometry_flagged"] for s in marked),
            "marked_unflagged": sum(not s["geometry_flagged"] for s in marked),
            "pending_cause_excluded_from_validation": sum(
                s["review_decision"] == "skipped" for s in states
            ),
            "runtime_table_issues_excluded_from_validation": sum(
                s["review_decision"] == "runtime_table_issue" for s in states
            ),
            "unreviewed_excluded_from_validation": sum(
                s["review_decision"] == "unreviewed" for s in states
            ),
            "note": (
                "The region rule screens separating contours; local small crossings are allowed. "
                "Sequential skips identify problematic trajectories, not the cause or onset. "
                "Only assessed initial self-intersections are positives, including hidden "
                "initial defects exposed by later actions. Runtime table issues and pending "
                "causes are neither initial-defect positives nor clean negatives. Marked "
                "trajectories are usable examples, not proof of intersection-free states."
            ),
        }
    if report_file.is_relative_to(root):
        raise ValueError("audit report must be outside the source collection")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(report_file, report)
    result = {
        key: value for key, value in report.items() if key not in {"assets", "sequential_review"}
    }
    result["report"] = str(report_file)
    if repair_output is not None:
        destination = Path(repair_output).expanduser().resolve()
        _repair_collection(root, destination, assets, meta, report, gpus)
        result["repair_output"] = str(destination)
        result["repaired_state_count"] = invalid_count
    return result


def _inspect_collection_asset(job: tuple[Path, Path, dict[str, Any]]) -> tuple[str, list[dict]]:
    root, assets, item = job
    asset_id = item["asset_id"]
    path = root / asset_id / "states.npz"
    pool_sha256 = _sha256(path)
    if item.get("sha256", pool_sha256) != pool_sha256:
        raise ValueError(f"initial-state pool hash differs from its collection: {path}")
    state_pool = InitialStatePool.load(path)
    states = inspect_pool_geometry(state_pool, GarmentLibrary(assets).converted(asset_id))
    return pool_sha256, states


def _repair_collection(root, destination, assets, meta, report, gpus) -> None:
    if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("repair output must be separate from the source collection")
    if not gpus or any(not gpu.strip() for gpu in gpus):
        raise ValueError("gpus must contain at least one device id")
    identity = {
        "source": str(root),
        "source_meta_sha256": report["source_meta_sha256"],
        "source_pool_sha256": {item["asset_id"]: item["sha256"] for item in report["assets"]},
        "geometry_validation": geometry_validation_settings(),
        "review_sha256": report["sequential_review"]["review_sha256"],
        "quality_review_sha256": report["sequential_review"]["quality_review_sha256"],
        "rollout_meta_sha256": report["sequential_review"]["rollout_meta_sha256"],
    }
    meta_path = destination / "meta.json"
    if meta_path.exists():
        previous = load_initial_state_collection(destination, require_complete=False)
        if previous.get("geometry_repair") != identity:
            raise ValueError("repair output belongs to a different source or validation policy")
    elif destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"repair output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    repaired_meta = {**meta, "status": "generating", "geometry_repair": identity}
    # A repaired panel needs a new benchmark freeze; never inherit a frozen identity.
    if "selection_provenance" in repaired_meta:
        repaired_meta["source_selection_provenance"] = repaired_meta.pop("selection_provenance")
    _atomic_write_json(meta_path, repaired_meta)
    by_id = {item["asset_id"]: item for item in meta["assets"]}
    replacement_root = destination / ".replacements"
    jobs = []
    for record in report["assets"]:
        if record["exclude_asset"]:
            continue
        rejected = [state for state in record["states"] if state["replace"]]
        if rejected and not (replacement_root / record["asset_id"] / "states.npz").exists():
            jobs.append(
                {
                    **by_id[record["asset_id"]],
                    "state_count": len(rejected),
                    "excluded_seeds": [state["seed"] for state in record["states"]],
                }
            )
    try:
        if jobs:
            _run_state_workers(
                jobs,
                gpus=gpus,
                assets_root=assets,
                output=replacement_root,
                states_per_asset=meta["states_per_asset"],
                candidate_batch=meta["candidate_batch"],
                max_candidates=meta["max_candidates"],
                drop_seconds=meta["drop_seconds"],
            )
        summaries = []
        for record in report["assets"]:
            if record["exclude_asset"]:
                continue
            asset_id = record["asset_id"]
            original_path = root / asset_id / "states.npz"
            if _sha256(original_path) != record["sha256"]:
                raise ValueError(f"source initial-state pool changed during repair: {asset_id}")
            target = destination / asset_id / "states.npz"
            original = InitialStatePool.load(original_path)
            bad = [state["index"] for state in record["states"] if state["replace"]]
            if not target.exists():
                if bad:
                    replacements = InitialStatePool.load(replacement_root / asset_id / "states.npz")
                    inspect_pool_geometry(replacements, GarmentLibrary(assets).converted(asset_id))
                    _merge_replacements(original, replacements, bad, record).save(target)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(".states.npz.tmp")
                    shutil.copyfile(original_path, temporary)
                    temporary.replace(target)
            merged = InitialStatePool.load(target)
            inspect_pool_geometry(merged, GarmentLibrary(assets).converted(asset_id))
            good = [state["index"] for state in record["states"] if not state["replace"]]
            if any(
                not np.array_equal(a, b)
                for a, b in zip(
                    merged.select(np.asarray(good)), original.select(np.asarray(good)), strict=True
                )
            ):
                raise ValueError(f"repair output changed a valid initial-state slot: {asset_id}")
            if merged.state_count != original.state_count:
                raise ValueError("repair output changed the number of initial states")
            summaries.append({**by_id[asset_id], "sha256": _sha256(target)})
        if _sha256(root / "meta.json") != report["source_meta_sha256"]:
            raise ValueError("source initial-state collection changed during repair")
        pending_review = [
            {
                "asset_id": record["asset_id"],
                "index": state["index"],
                "reason": "replacement" if state["replace"] else "cause_unconfirmed",
            }
            for record in report["assets"]
            for state in record["states"]
            if state["replace"] or state["review_decision"] == "skipped"
        ]
        repaired_meta.update(
            status="pending_review" if pending_review else "complete",
            assets=summaries,
            seen_assets=sum(item["split"] == "seen" for item in summaries),
            unseen_assets=sum(item["split"] == "unseen" for item in summaries),
            excluded_assets=[r["asset_id"] for r in report["assets"] if r["exclude_asset"]],
            pending_review=pending_review,
        )
        repaired_meta.pop("error", None)
        _atomic_write_json(meta_path, repaired_meta)
    except Exception as exc:
        repaired_meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _atomic_write_json(meta_path, repaired_meta)
        raise


def _merge_replacements(original, replacements, bad, record) -> InitialStatePool:
    if replacements.state_count != len(bad):
        raise ValueError("replacement count differs from rejected slot count")
    if set(map(int, replacements.seeds)) & set(map(int, original.seeds)):
        raise ValueError("replacement seeds must not reuse original state seeds")
    q, qd, seeds = original.select(np.arange(original.state_count))
    q[bad] = replacements.particle_q
    qd[bad] = replacements.particle_qd
    seeds[bad] = replacements.seeds
    metadata = dict(original.metadata)
    metrics = [dict(item) for item in metadata.get("accepted_metrics", [{}] * len(seeds))]
    for state in record["states"]:
        metrics[state["index"]].update(
            {key: value for key, value in state.items() if key.startswith("self_intersection_")}
        )
    for offset, index in enumerate(bad):
        metrics[index] = replacements.metadata["accepted_metrics"][offset]
    metadata.update(
        accepted_metrics=metrics,
        # Retained slots were audited, not regenerated under this acceptance rule.
        # The replacement sub-pool records its own generation policy below.
        geometry_diagnostics={**geometry_validation_settings(), "automatic_rejection": False},
        geometry_repair={
            "source_pool_sha256": record["sha256"],
            "replaced_indices": bad,
            "original_seeds": [int(original.seeds[index]) for index in bad],
            "replacement_generation": replacements.metadata,
        },
    )
    return InitialStatePool(q, qd, seeds, metadata)
