"""Generate four settled folding starts from a garment's flat reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def quarter_turns(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
    relative = points[:, :2] - center
    result = []
    for cosine, sine in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        rotated = points.copy()
        rotated[:, 0] = center[0] + cosine * relative[:, 0] - sine * relative[:, 1]
        rotated[:, 1] = center[1] + sine * relative[:, 0] + cosine * relative[:, 1]
        result.append(rotated)
    return np.ascontiguousarray(result, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--garment", default="rand_ZZCWITH5HY")
    parser.add_argument("--template", type=Path,
                        help="default: templates/<garment>.npz")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path,
                        help="default: outputs/initial-states/<garment>-fold.npz")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.template is None:
        args.template = Path("templates") / f"{args.garment}.npz"
    if args.output is None:
        args.output = Path("outputs/initial-states") / f"{args.garment}-fold.npz"
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    from clothdojo.assets import GarmentLibrary
    from clothdojo.evaluation.metric import load_canonical_template
    from clothdojo.rollout.initial_states import InitialStatePool, validate_pool_compatibility
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

    assets = args.assets.resolve()
    garment = GarmentLibrary(assets).converted(args.garment)
    template = load_canonical_template(args.template)
    garment_sha = hashlib.sha256(garment.read_bytes()).hexdigest()
    if template.asset_id != args.garment or template.asset_sha256 != garment_sha:
        parser.error("template does not match the garment")
    spec = ShirtFlattenSpec(garment_id=args.garment, world_count=4)
    rotations = quarter_turns(template.particle_q)
    scene = build_scene(assets, spec=spec, task_name="shirt_fold", device=args.device)
    runtime = ShirtFlattenRuntime(scene)
    table_center = np.asarray(spec.table_position[:2], dtype=np.float64)
    table_half = np.asarray(spec.table_half_extents[:2], dtype=np.float64)
    table_top = spec.table_position[2] + spec.table_half_extents[2]
    ranges = runtime.grasp.particle_ranges()
    offsets = np.asarray(scene.world_offsets, dtype=np.float32)[:, None, :]
    try:
        for shift_cm in range(51):
            placed = rotations.copy()
            placed[:, :, 0] += shift_cm / 100
            runtime.restore_particles(placed, np.zeros_like(placed))
            _, robot = runtime.cloth_contact_counts()
            if np.any(robot):
                continue
            graph = None
            for tick in range(300):
                graph = runtime.advance_cloth(5, graph=graph)
                q_all = runtime.state_0.particle_q.numpy()
                qd_all = runtime.state_0.particle_qd.numpy()
                q = np.ascontiguousarray(np.stack([
                    q_all[start:end] - offsets[index]
                    for index, (start, end) in enumerate(ranges)
                ]), dtype=np.float32)
                qd = np.ascontiguousarray(np.stack([
                    qd_all[start:end] for start, end in ranges
                ]), dtype=np.float32)
                speed = np.sqrt(np.mean(np.sum(qd.astype(np.float64) ** 2, axis=2), axis=1))
                if tick >= 79 and np.all(speed <= 0.02):
                    break
            table, robot = runtime.cloth_contact_counts()
            inside = np.all(np.abs(q[:, :, :2] - table_center) <= table_half - 0.01)
            valid = (np.isfinite(q).all() and np.isfinite(qd).all()
                     and np.all(speed <= 0.02) and np.all(table > 0)
                     and not np.any(robot) and inside
                     and np.all(q[:, :, 2] >= table_top - 0.001))
            if valid:
                break
        else:
            raise RuntimeError("no valid settled four-heading fold pool within 0.50 m shift")
        pool = InitialStatePool(q, qd, np.arange(4, dtype=np.int32), {
            "garment_id": args.garment,
            "garment_sha256": garment_sha,
            "garment_scale": spec.garment_scale,
            "task_name": "shirt_fold",
            "initialization": "settled_flat_quarter_turns_v1",
            "angles_deg": [0, 90, 180, 270],
            "translation_x_m": shift_cm / 100,
            "template_sha256": hashlib.sha256(args.template.read_bytes()).hexdigest(),
            "speed_rms_m_s": speed.tolist(),
            "table_contacts": table.tolist(),
        })
        pool.save(args.output)
        validate_pool_compatibility(
            InitialStatePool.load(args.output), garment_path=garment,
            particle_count=scene.model.particle_count // 4,
        )
        print(json.dumps({"output": str(args.output), "states": 4,
                          "shift_m": shift_cm / 100}, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
