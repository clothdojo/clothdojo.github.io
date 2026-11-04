"""Four deterministic fold starts from a standard settled garment."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from clothdojo.rollout.initial_states import (
    InitialStatePool,
    _atomic_write_json,
    _capture_particles,
    _json_sha256,
    _sha256,
    validate_initial_states,
)

FOLD_INITIALIZATION = "standard_center_quarter_turns_v1"
FOLD_ANGLES_DEG = (0, 90, 180, 270)


def quarter_turn_states(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate about the garment XY bounding-box center, preserving height and scale."""
    q = np.asarray(points, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
        raise ValueError("standard garment must contain finite XYZ vertices")
    center = (q[:, :2].min(axis=0) + q[:, :2].max(axis=0)) * 0.5
    xy = q[:, :2] - center
    rotations = []
    for cosine, sine in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        rotated = q.copy()
        rotated[:, 0] = center[0] + cosine * xy[:, 0] - sine * xy[:, 1]
        rotated[:, 1] = center[1] + sine * xy[:, 0] + cosine * xy[:, 1]
        rotations.append(rotated)
    return np.ascontiguousarray(rotations, dtype=np.float32), center


def generate_fold_initial_states(
    assets_root: str | Path,
    output: str | Path,
    *,
    garment_id: str,
    protocol_path: str | Path,
    device: str = "cuda:0",
) -> dict:
    """Save all four angles only when every angle passes the same quality checks."""
    from clothdojo.assets import GarmentLibrary
    from clothdojo.evaluation.metric import factorized_cloth_distance, load_canonical_template
    from clothdojo.evaluation.protocol import load_evaluation_protocol
    from clothdojo.evaluation.templates import generate_asset_template, validate_template_physics
    from clothdojo.rollout.state_geometry import geometry_diagnostics_settings
    from clothdojo.sim.cloth import load_garment_mesh
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    assets = Path(assets_root).resolve()
    garment = GarmentLibrary(assets).converted(garment_id)
    spec = ShirtFlattenSpec(garment_id=garment_id, world_count=4)
    protocol = load_evaluation_protocol(protocol_path)
    reference_path = destination.parent / "standard.npz"
    if not reference_path.exists():
        result = generate_asset_template(
            garment_id, protocol_path, reference_path, assets_root=assets, device=device
        )
        _atomic_write_json(destination.parent / "standard.json", result)
    template = load_canonical_template(reference_path)
    if template.asset_id != garment_id or template.asset_sha256 != _sha256(garment):
        raise ValueError("standard reference garment identity differs")
    validate_template_physics(template, replace(spec, world_count=1))
    rest, indices, _, _ = load_garment_mesh(garment)
    rotations, center = quarter_turn_states(template.particle_q)
    scene = build_scene(assets, task_name="shirt_fold", spec=spec, device=device)
    runtime = ShirtFlattenRuntime(scene)
    ranges = runtime.grasp.particle_ranges()
    offsets = np.asarray(scene.world_offsets, dtype=np.float32)[:, None, :]
    # Piper's base is on the low-X side of the table. Search one common +X
    # translation for all headings, without deforming or rescaling the garment.
    graph = None
    for translation_x in np.arange(51, dtype=np.float64) * 0.01:
        placed = rotations.copy()
        placed[:, :, 0] += translation_x
        runtime.restore_particles(placed, np.zeros_like(placed))
        _, initial_robot = runtime.cloth_contact_counts()
        if np.any(initial_robot):
            continue
        graph = runtime.advance_cloth(spec.control_fps, graph=graph)
        tail = deque(maxlen=10)
        steps = spec.control_fps
        while steps < protocol.target_max_settle_steps:
            graph = runtime.advance_cloth(spec.control_fps // 10, graph=graph)
            steps += spec.control_fps // 10
            q, qd = _capture_particles(runtime, ranges)
            q -= offsets
            tail.append(np.sqrt(np.mean(np.sum(qd.astype(np.float64)**2, axis=2), axis=1)))
            if len(tail) == tail.maxlen and np.max(tail) <= protocol.target.speed_rms_max_m_s:
                break
        table, robot = runtime.cloth_contact_counts()
        if not np.any(robot):
            break
    else:
        raise RuntimeError("no common robot-free placement within 0.50 m along +X")
    validation = validate_initial_states(
        q, qd, table_top_m=spec.table_position[2] + spec.table_half_extents[2],
        maximum_speed_rms_m_s=protocol.target.speed_rms_max_m_s,
        table_bounds_xy_m=(
            tuple(np.asarray(spec.table_position[:2]) - spec.table_half_extents[:2]),
            tuple(np.asarray(spec.table_position[:2]) + spec.table_half_extents[:2]),
        ),
        table_contact_count=table, robot_contact_count=robot,
        triangles=indices.reshape(-1, 3), rest_points=rest * spec.garment_scale,
    )
    tail_max = np.max(np.stack(tail), axis=0)
    records = []
    for i, angle in enumerate(FOLD_ANGLES_DEG):
        distance = factorized_cloth_distance(
            q[i:i+1], qd[i:i+1], replace(template, particle_q=placed[i]), symmetry="none"
        )
        reasons = list(validation.reasons[i])
        if tail_max[i] > protocol.target.speed_rms_max_m_s:
            reasons.append("unstable_tail")
        if distance.deformable_distance[0] > protocol.deformable_distance_max:
            reasons.append("standard_shape_changed")
        if distance.rigid_distance[0] > protocol.rigid_distance_max:
            reasons.append("standard_heading_changed")
        records.append({
            "index": i, "angle_deg": angle, "valid": not reasons, "reasons": reasons,
            "self_intersection_score": float(validation.self_intersection_score[i]),
            "table_contact_count": int(table[i]), "robot_contact_count": int(robot[i]),
            "speed_rms_m_s": float(validation.speed_rms_m_s[i]),
            "tail_speed_rms_max_m_s": float(tail_max[i]),
            "flat_deformable_distance": float(distance.deformable_distance[0]),
            "flat_heading_distance": float(distance.rigid_distance[0]),
        })
    metadata = {
        "garment_id": garment_id, "garment_path": str(garment), "garment_sha256": _sha256(garment),
        "garment_scale": spec.garment_scale, "task_name": "shirt_fold", "robot_id": "piper",
        "scene_sha256": _json_sha256(asdict(replace(spec, world_count=1))),
        "initialization": FOLD_INITIALIZATION, "angles_deg": list(FOLD_ANGLES_DEG),
        "rotation_center_xy": center.tolist(), "rotation_center": "standard_garment_xy_bbox_center",
        "translation_xyz_m": [float(translation_x), 0.0, 0.0],
        "placement_center_xy": [float(center[0] + translation_x), float(center[1])],
        "placement_method": "common_positive_x_first_contact_free_1cm_step_max_50cm",
        "standard_sha256": _sha256(reference_path), "standard_source": "default_free_settle",
        "post_rotation_settle_seconds": steps/spec.control_fps,
        "geometry_diagnostics": geometry_diagnostics_settings(), "accepted_metrics": records,
        "flatness_protocol_fingerprint": protocol.fingerprint,
        "flatness_deformable_max": protocol.deformable_distance_max,
        "flatness_heading_max": protocol.rigid_distance_max,
        "seed_semantics": "deterministic quarter-turn index; no random sampling",
    }
    _atomic_write_json(destination.parent / "validation.json", metadata)
    if not all(r["valid"] for r in records):
        np.savez_compressed(destination.parent / "rejected.npz", particle_q=q, particle_qd=qd)
        failures = [(r["angle_deg"], r["reasons"]) for r in records if not r["valid"]]
        raise RuntimeError(f"invalid fold angles: {failures}")
    InitialStatePool(q, qd, np.arange(4, dtype=np.int32), metadata).save(destination)
    return {"asset_id": garment_id, "state_count": 4, "particle_count": len(q[0]),
            "sha256": _sha256(destination), "path": f"{garment_id}/states.npz"}
