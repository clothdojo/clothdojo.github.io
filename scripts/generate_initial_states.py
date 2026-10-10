"""Generate a garment-specific flattening initial-state pool by hang and drop."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np


def rotation_from_seed(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    x, y, z, w = quaternion
    return np.asarray(
        ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
         (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
         (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        dtype=np.float64,
    )


def state_quality(q: np.ndarray, qd: np.ndarray, table_top: float,
                  table_contacts: int, robot_contacts: int) -> tuple[bool, dict]:
    height = q[:, 2] - table_top
    p99 = float(np.quantile(height, 0.99))
    elevated = float(np.mean(height > 0.40))
    near_table = float(np.mean(height <= 0.03))
    speed = float(np.sqrt(np.mean(np.sum(qd * qd, axis=1))))
    reasons = []
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        reasons.append("non_finite")
    if p99 > 0.25:
        reasons.append("too_high")
    if elevated > 0.002:
        reasons.append("elevated")
    if near_table < 0.01 or table_contacts < 1:
        reasons.append("no_table_contact")
    if speed > 0.20:
        reasons.append("unsettled")
    if robot_contacts:
        reasons.append("robot_contact")
    return not reasons, {
        "p99_height_above_table_m": p99,
        "elevated_fraction": elevated,
        "near_table_fraction": near_table,
        "speed_rms_m_s": speed,
        "table_contacts": int(table_contacts),
        "robot_contacts": int(robot_contacts),
        "rejection_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--garment", default="rand_ZZCWITH5HY")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path,
                        help="default: outputs/initial-states/<garment>-flatten.npz")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260815)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--drop-seconds", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("outputs/initial-states") / f"{args.garment}-flatten.npz"
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if min(args.count, args.batch, args.max_candidates) < 1 or args.max_candidates < args.count:
        parser.error("count, batch and max-candidates must be positive; max-candidates >= count")
    if args.max_candidates % args.batch:
        parser.error("max-candidates must be divisible by batch")
    if not 0 <= args.seed <= np.iinfo(np.int32).max:
        parser.error("seed must be a non-negative int32")
    if args.hold_seconds <= 0 or args.drop_seconds <= 0:
        parser.error("hold-seconds and drop-seconds must be positive")

    from clothdojo.assets import GarmentLibrary
    from clothdojo.rollout.initial_states import InitialStatePool, validate_pool_compatibility
    from clothdojo.sim.grasp import StaticParticlePatch
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

    assets = args.assets.resolve()
    garment = GarmentLibrary(assets).converted(args.garment)
    spec = ShirtFlattenSpec(garment_id=args.garment, world_count=args.batch)
    scene = build_scene(assets, spec=spec, device=args.device)
    runtime = ShirtFlattenRuntime(scene)
    accepted_q, accepted_qd, accepted_seeds, metrics = [], [], [], []
    rejected = Counter()
    generator = np.random.default_rng(args.seed)
    used_seeds = set()
    candidate_count = 0
    table_top = spec.table_position[2] + spec.table_half_extents[2]
    try:
        ranges = runtime.grasp.particle_ranges()
        patch = StaticParticlePatch(scene)
        graph = None
        while len(accepted_q) < args.count and candidate_count < args.max_candidates:
            seeds = []
            while len(seeds) < args.batch:
                value = int(generator.integers(0, np.iinfo(np.int32).max))
                if value not in used_seeds:
                    seeds.append(value)
                    used_seeds.add(value)
            runtime.reset()
            base = runtime.state_0.particle_q.numpy()
            positions, patches = [], []
            for world, ((start, end), seed) in enumerate(zip(ranges, seeds, strict=True)):
                points = np.asarray(base[start:end], dtype=np.float64) - scene.world_offsets[world]
                rotated = (points - points.mean(axis=0)) @ rotation_from_seed(seed).T
                pin_local = int(np.argmax(rotated[:, 2]))
                diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
                anchor = np.asarray((points[:, 0].mean(), points[:, 1].mean(),
                                     table_top + diagonal + 0.05))
                rotated += anchor - rotated[pin_local]
                positions.append(rotated.astype(np.float32))
                patches.append(runtime.grasp.connected_patch(start + pin_local))
            q = np.ascontiguousarray(np.stack(positions), dtype=np.float32)
            runtime.restore_particles(q, np.zeros_like(q))
            patch.configure(patches, runtime.state_0.particle_q)
            graph = runtime.advance_cloth(round(args.hold_seconds * spec.control_fps),
                                          graph=graph, static_patch=patch)
            patch.disable()
            runtime.cloth.rebuild_bvh(runtime.state_0)
            graph = runtime.advance_cloth(round(args.drop_seconds * spec.control_fps), graph=graph)
            all_q = runtime.state_0.particle_q.numpy()
            all_qd = runtime.state_0.particle_qd.numpy()
            table_contacts, robot_contacts = runtime.cloth_contact_counts()
            for world, ((start, end), seed) in enumerate(zip(ranges, seeds, strict=True)):
                local_q = np.ascontiguousarray(all_q[start:end] - scene.world_offsets[world],
                                               dtype=np.float32)
                local_qd = np.ascontiguousarray(all_qd[start:end], dtype=np.float32)
                valid, quality = state_quality(local_q, local_qd, table_top,
                                               table_contacts[world], robot_contacts[world])
                candidate_count += 1
                if valid and len(accepted_q) < args.count:
                    accepted_q.append(local_q)
                    accepted_qd.append(local_qd)
                    accepted_seeds.append(seed)
                    metrics.append(quality)
                else:
                    rejected.update(quality["rejection_reasons"] or ["surplus"])
            print(f"accepted {len(accepted_q)}/{args.count} after {candidate_count} candidates",
                  flush=True)
        if len(accepted_q) != args.count:
            raise RuntimeError(f"only {len(accepted_q)}/{args.count} states passed after "
                               f"{candidate_count} candidates; rejected={dict(rejected)}")
        garment_sha = hashlib.sha256(garment.read_bytes()).hexdigest()
        metadata = {
            "garment_id": args.garment,
            "garment_sha256": garment_sha,
            "garment_scale": spec.garment_scale,
            "task_name": "shirt_flatten",
            "initialization": "random_hang_drop_v1",
            "generator_seed": args.seed,
            "hold_seconds": args.hold_seconds,
            "drop_seconds": args.drop_seconds,
            "candidate_count": candidate_count,
            "rejection_counts": dict(rejected),
            "accepted_metrics": metrics,
            "scene_sha256": hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest(),
        }
        pool = InitialStatePool(
            np.ascontiguousarray(np.stack(accepted_q)),
            np.ascontiguousarray(np.stack(accepted_qd)),
            np.asarray(accepted_seeds, dtype=np.int32), metadata,
        )
        pool.save(args.output)
        loaded = InitialStatePool.load(args.output)
        validate_pool_compatibility(loaded, garment_path=garment,
                                    particle_count=scene.model.particle_count // args.batch)
        print(json.dumps({"output": str(args.output), "states": loaded.state_count,
                          "candidates": candidate_count, "sha256": hashlib.sha256(
                              args.output.read_bytes()).hexdigest()}, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
