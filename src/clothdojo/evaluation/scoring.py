"""Score a recorded ClothDojo trajectory with its task's fixed protocol."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from clothdojo.evaluation.fold_quality import FoldQualityMetric, prepare_fold_quality_reference
from clothdojo.evaluation.fold_success import fold_success
from clothdojo.evaluation.metric import (
    factorized_cloth_distance,
    file_sha256,
    load_canonical_template,
)
from clothdojo.evaluation.protocol import SuccessTracker, load_evaluation_protocol
from clothdojo.evaluation.templates import validate_template_physics
from clothdojo.trajectory import TrajectoryReader


def score_trajectory(
    trajectory: str | Path,
    template_path: str | Path,
    *,
    assets: str | Path = "assets",
    protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the same task score used by the standalone scoring scripts."""

    from clothdojo.sim.scene import ShirtFlattenSpec

    reader = TrajectoryReader(trajectory)
    reader.validate_render_data()
    task = reader.manifest.scene.task_name
    if task not in {"shirt_flatten", "shirt_fold"}:
        raise ValueError(f"unsupported scoring task: {task}")
    protocol = load_evaluation_protocol(
        protocol_path or Path(f"configs/evaluation/{task}.yaml")
    )
    template = load_canonical_template(template_path)
    if protocol.is_fold != (task == "shirt_fold"):
        raise ValueError("trajectory and evaluation protocol tasks differ")
    if reader.manifest.control_fps != protocol.control_fps:
        raise ValueError("trajectory frame rate differs from the protocol")
    metadata = reader.manifest.metadata
    if "simulation_spec" not in metadata:
        raise ValueError("trajectory must record simulation_spec")
    final_step = min(reader.manifest.action_count, protocol.horizon_steps)
    positions = reader.state_arrays(["particle_q"])["particle_q"]
    complete = final_step == protocol.horizon_steps

    if task == "shirt_flatten":
        if (metadata.get("asset_id") != template.asset_id
                or reader.manifest.scene.asset_fingerprints.get("garment_usdc") != template.asset_sha256):
            raise ValueError("trajectory and template garment identities differ")
        validate_template_physics(template, ShirtFlattenSpec(**metadata["simulation_spec"]))
        tracker = SuccessTracker(protocol, batch_size=1)
        last = None
        for step in range(0, final_step + 1, protocol.metric_stride):
            q = positions[step:step + 1]
            last = factorized_cloth_distance(
                q, np.zeros_like(q), template,
                deformable_weight=protocol.deformable_weight,
                symmetry=protocol.symmetry,
                symmetry_center_xy=protocol.target.symmetry_center_xy,
            )
            tracker.update(
                step=step,
                deformable_distance=last.deformable_distance,
                rigid_distance=last.rigid_distance,
            )
        confirmed = bool(tracker.success[0])
        return {
            "protocol_id": protocol.protocol_id,
            "protocol_fingerprint": protocol.fingerprint,
            "evaluated_steps": final_step,
            "horizon_complete": complete,
            "success": True if confirmed else (False if complete else None),
            "confirmed_step": int(tracker.confirmed_step[0]) if confirmed else None,
            "last_deformable_distance": float(last.deformable_distance[0]),
            "last_rigid_distance": float(last.rigid_distance[0]),
        }

    from clothdojo.sim.cloth import load_garment_mesh

    garment = Path(assets) / "garments/converted" / template.asset_id / f"{template.asset_id}.usdc"
    if (metadata.get("asset_id") != template.asset_id
            or file_sha256(garment) != template.asset_sha256
            or reader.manifest.scene.asset_fingerprints.get("garment_usdc") != template.asset_sha256):
        raise ValueError("trajectory, template and garment identities differ")
    validate_template_physics(template, ShirtFlattenSpec(**metadata["simulation_spec"]))
    triangles = load_garment_mesh(garment)[1].reshape(-1, 3)
    reference = prepare_fold_quality_reference(template.particle_q, triangles)
    metric = FoldQualityMetric.load()
    result = fold_success(
        final_step + 1, protocol.control_fps,
        lambda step: float(metric.score(positions[step], reference)),
    )
    return {
        **asdict(result),
        "success": True if result.success else (False if complete else None),
        "horizon_complete": complete,
        "evaluated_steps": final_step,
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "terminal_q": float(metric.score(positions[final_step], reference)),
        "threshold": metric.success_threshold,
    }
