"""Canonical free-settle targets and physics identity checks."""
from __future__ import annotations
import hashlib
import json
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
import numpy as np
from clothdojo.evaluation.metric import CanonicalTemplate, file_sha256, save_canonical_template
LARGE_SUPPORT_METHOD = "large_support_free_settle_v1"
LARGE_SUPPORT_HALF_EXTENT_M = 2.0
TARGET_EDGE_CLEARANCE_M = 0.10
TARGET_HEIGHT_TOLERANCE_M = 0.001


def canonical_target_spec(spec, method: str):
    """Resolve the versioned target support without changing the rollout scene."""
    if method == "default_free_settle":
        return spec
    if method != LARGE_SUPPORT_METHOD:
        raise ValueError("unsupported canonical target method")
    return replace(spec, table_half_extents=(
        LARGE_SUPPORT_HALF_EXTENT_M, LARGE_SUPPORT_HALF_EXTENT_M,
        spec.table_half_extents[2],
    ))


def validate_target_support(particle_q, spec) -> dict[str, Any]:
    """Reject edge-dependent targets and particles hanging below the support."""
    q = np.asarray(particle_q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
        raise ValueError("canonical target particles must be finite (N, 3)")
    clearance = np.asarray(spec.table_half_extents[:2]) - np.abs(
        q[:, :2] - np.asarray(spec.table_position[:2])
    )
    top = spec.table_position[2] + spec.table_half_extents[2]
    min_clearance = float(clearance.min())
    min_height = float(q[:, 2].min() - top)
    if min_clearance < TARGET_EDGE_CLEARANCE_M:
        raise ValueError(f"canonical target is too close to support edge: {min_clearance:.6f} m")
    if min_height < -TARGET_HEIGHT_TOLERANCE_M:
        raise ValueError(f"canonical target hangs below support: {min_height:.6f} m")
    return {
        "table_position_m": list(spec.table_position),
        "table_half_extents_m": list(spec.table_half_extents),
        "minimum_edge_clearance_m": min_clearance,
        "minimum_height_above_support_m": min_height,
        "required_edge_clearance_m": TARGET_EDGE_CLEARANCE_M,
        "height_tolerance_m": TARGET_HEIGHT_TOLERANCE_M,
    }


def generate_asset_template(
    asset_id: str,
    protocol_path: str | Path,
    output: str | Path,
    *,
    assets_root: str | Path = "assets",
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Free-settle the default pose on the protocol's robot-independent support."""

    from clothdojo.assets import GarmentLibrary
    from clothdojo.evaluation.protocol import load_evaluation_protocol
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

    if not asset_id:
        raise ValueError("asset_id must not be empty")
    protocol = load_evaluation_protocol(protocol_path)
    target_protocol = protocol.target
    if target_protocol.robot_particle_collision:
        raise ValueError("canonical targets must disable robot particle collision")
    assets = Path(assets_root).expanduser().resolve()
    garment = GarmentLibrary(assets).converted(asset_id)
    spec = ShirtFlattenSpec(garment_id=asset_id, world_count=1)
    spec = canonical_target_spec(spec, target_protocol.method)
    if spec.control_fps != protocol.control_fps:
        raise ValueError("target protocol and scene control_fps differ")
    if not np.array_equal(
        np.asarray(target_protocol.symmetry_center_xy, dtype=np.float64),
        np.asarray(spec.table_position[:2], dtype=np.float64),
    ):
        raise ValueError("target symmetry center differs from the scene table center")

    scene = build_scene(
        assets,
        spec=spec,
        device=device,
        visuals=False,
        robot_cloth_collision=False,
    )
    runtime = ShirtFlattenRuntime(scene, use_cuda_graph=False)
    try:
        if target_protocol.method == LARGE_SUPPORT_METHOD:
            validate_target_support(runtime.state_0.particle_q.numpy(), spec)
        ranges = runtime.grasp.particle_ranges()
        if len(ranges) != 1:
            raise RuntimeError("canonical target generation requires one Newton world")
        start, end = ranges[0]
        min_steps = protocol.target_min_settle_steps
        max_steps = protocol.target_max_settle_steps
        sample_stride = protocol.metric_stride
        window_samples = protocol.target_stability_samples
        window_steps = window_samples * sample_stride
        graph = runtime.advance_cloth(min_steps - window_steps)
        tail_speeds: deque[float] = deque(maxlen=window_samples)
        settled_steps = min_steps - window_steps
        while settled_steps < max_steps:
            graph = runtime.advance_cloth(sample_stride, graph=graph)
            qd = runtime.state_0.particle_qd.numpy()[start:end]
            tail_speeds.append(_speed_rms(qd))
            settled_steps += sample_stride
            if (
                settled_steps >= min_steps
                and len(tail_speeds) == window_samples
                and max(tail_speeds) <= target_protocol.speed_rms_max_m_s
            ):
                break

        particle_q = np.ascontiguousarray(
            runtime.state_0.particle_q.numpy()[start:end], dtype=np.float32
        )
        particle_qd = np.ascontiguousarray(
            runtime.state_0.particle_qd.numpy()[start:end], dtype=np.float32
        )
        if not np.isfinite(particle_q).all() or not np.isfinite(particle_qd).all():
            raise RuntimeError("canonical target contains non-finite particle state")
        maximum_tail_speed = float(max(tail_speeds))
        if maximum_tail_speed > target_protocol.speed_rms_max_m_s:
            raise RuntimeError(
                "canonical target did not settle before timeout: "
                f"tail_speed_max={maximum_tail_speed:.8f} > "
                f"{target_protocol.speed_rms_max_m_s:.8f} after "
                f"{settled_steps / spec.control_fps:.2f}s"
            )
        table_contacts, robot_contacts = runtime.cloth_contact_counts()
        if int(table_contacts[0]) < 1:
            raise RuntimeError("canonical target has no physical table contact")
        if int(robot_contacts[0]) != 0:
            raise RuntimeError("canonical target unexpectedly contacted the ghosted robot")
        extent = np.ptp(particle_q[:, :2].astype(np.float64), axis=0)
        support = (
            validate_target_support(particle_q, spec)
            if target_protocol.method == LARGE_SUPPORT_METHOD else None
        )
        scale = float(np.linalg.norm(extent))
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("canonical target has a degenerate planar extent")

        asset_sha256 = file_sha256(garment)
        topology_sha256 = garment_topology_sha256(garment)
        physics_sha256 = shirt_flatten_physics_sha256(spec)
        source_sha256 = _target_source_sha256(
            asset_sha256=asset_sha256,
            topology_sha256=topology_sha256,
            physics_sha256=physics_sha256,
            protocol_fingerprint=protocol.fingerprint,
            particle_q=particle_q,
        )
        template = CanonicalTemplate(
            asset_id=asset_id,
            asset_sha256=asset_sha256,
            topology_sha256=topology_sha256,
            scene_physics_sha256=physics_sha256,
            source_kind=target_protocol.method,
            source_sha256=source_sha256,
            normalization_scale_m=scale,
            particle_q=particle_q,
        )
        destination = Path(output).expanduser().resolve()
        save_canonical_template(destination, template)
        table_top = spec.table_position[2] + spec.table_half_extents[2]
        return {
            "asset_id": asset_id,
            "path": f"templates/{asset_id}.npz",
            "sha256": file_sha256(destination),
            "protocol_fingerprint": protocol.fingerprint,
            "particle_count": template.particle_count,
            "normalization_scale_m": template.normalization_scale_m,
            "tail_speed_rms_max_m_s": maximum_tail_speed,
            "settled_seconds": settled_steps / spec.control_fps,
            "terminal_speed_rms_m_s": _speed_rms(particle_qd),
            "table_contact_count": int(table_contacts[0]),
            "p99_height_above_table_m": float(
                np.quantile(particle_q[:, 2].astype(np.float64), 0.99) - table_top
            ),
            "source_kind": template.source_kind,
            "source_sha256": template.source_sha256,
            **({"support_validation": support} if support is not None else {}),
        }
    finally:
        runtime.close()


def garment_topology_sha256(path: str | Path) -> str:
    """Fingerprint the particle correspondence and triangle order."""

    from clothdojo.sim.cloth import load_garment_mesh

    vertices, indices, _, _ = load_garment_mesh(path)
    triangles = np.asarray(indices, dtype="<i4", order="C")
    payload = np.asarray([len(vertices), len(triangles)], dtype="<i8").tobytes()
    payload += triangles.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def shirt_flatten_physics_sha256(spec) -> str:
    """Fingerprint target-relevant physics, excluding batch and asset identity."""

    payload = asdict(spec)
    payload.pop("garment_id")
    payload.pop("world_count")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_template_physics(template: CanonicalTemplate, spec) -> None:
    """Validate runtime physics with the explicitly versioned target support."""
    if template.source_kind not in {"default_free_settle", LARGE_SUPPORT_METHOD}:
        raise ValueError("canonical template physics differs from the runtime scene")
    spec = canonical_target_spec(spec, template.source_kind)
    if template.scene_physics_sha256 == shirt_flatten_physics_sha256(spec):
        if template.source_kind == LARGE_SUPPORT_METHOD:
            validate_target_support(template.particle_q, spec)
        return
    # The frozen balanced-200 templates included ShirtFlattenSpec.robot_friction
    # (0.6). That field moved to the robot USD. Free-settle targets disable robot
    # particle contacts, so this field never influenced those cloth targets.
    # Recompute from the CURRENT spec: a changed cloth/table setting still fails.
    if template.source_kind == "default_free_settle":
        payload = asdict(spec)
        payload.pop("garment_id")
        payload.pop("world_count")
        payload["robot_friction"] = 0.6
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if template.scene_physics_sha256 == hashlib.sha256(encoded).hexdigest():
            return
    raise ValueError("canonical template physics differs from the runtime scene")


def _target_source_sha256(
    *, asset_sha256, topology_sha256, physics_sha256, protocol_fingerprint, particle_q
):
    digest = hashlib.sha256()
    identity = {
        "asset_sha256": asset_sha256,
        "topology_sha256": topology_sha256,
        "physics_sha256": physics_sha256,
        "protocol_fingerprint": protocol_fingerprint,
    }
    digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    digest.update(np.asarray(particle_q, dtype="<f4", order="C").tobytes())
    return digest.hexdigest()


def _speed_rms(particle_qd) -> float:
    values = np.asarray(particle_qd, dtype=np.float64)
    return float(np.sqrt(np.square(values).sum(axis=1).mean()))
