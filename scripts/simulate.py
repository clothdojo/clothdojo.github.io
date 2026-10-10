"""Step the Piper cloth scene and validate its saved trajectory."""

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, default=Path("outputs/simulation"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("simulation requires Linux x86_64; use score_fold.py for the CPU example")

    from clothdojo.assets.bundle import verify_simulation_assets

    verify_simulation_assets(args.assets)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    import warp as wp

    from clothdojo.assets import GarmentLibrary
    from clothdojo.evaluation.metric import CanonicalTemplate, factorized_cloth_distance, file_sha256
    from clothdojo.sim.cloth import load_garment_mesh
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene
    from clothdojo.trajectory import RUNTIME_ACTION_SEMANTICS, RenderOnlyTrajectoryWriter, SceneMetadata, TrajectoryReader
    from clothdojo.trajectory.newton import capture_canonical_actions, capture_states

    spec = ShirtFlattenSpec()
    garment = GarmentLibrary(args.assets).converted(spec.garment_id)
    topology = load_garment_mesh(garment)[1]
    scene = build_scene(args.assets, spec=spec, device=args.device)
    runtime = ShirtFlattenRuntime(scene)
    try:
        initial = capture_states(runtime, 0)[0]
        profile = scene.robot_profile
        arms = wp.array(
            np.asarray(profile.home_joint_q, dtype=np.float32)[list(profile.arm_coord_indices)][None],
            dtype=wp.float32, device=args.device,
        )
        grippers = wp.ones((1, 2), dtype=wp.float32, device=args.device)
        with RenderOnlyTrajectoryWriter(
            args.output, control_fps=spec.control_fps, sim_substeps=spec.sim_substeps,
            max_actions=args.steps, scene=SceneMetadata(task_name="shirt_flatten", robot_id="piper"),
            metadata={"training_eligible": False, "asset_id": spec.garment_id,
                      "action_semantics": RUNTIME_ACTION_SEMANTICS,
                      "purpose": "fixed_action_simulation_check", "simulation_spec": asdict(spec)},
        ) as writer:
            writer.write_initial_state(initial)
            for step in range(args.steps):
                runtime.step(arms, grippers)
                writer.append_transition(
                    capture_canonical_actions(runtime, step)[0],
                    capture_states(runtime, step + 1)[0],
                )
        reader = TrajectoryReader(args.output)
        reader.validate_render_data()
        positions = reader.state_arrays(["particle_q"])["particle_q"]
        if reader.manifest.state_count != args.steps + 1 or reader.manifest.action_count != args.steps:
            raise RuntimeError("saved trajectory length does not match the requested steps")
        # This initial-state reference measures motion, not task completion.
        reference = CanonicalTemplate(
            asset_id=spec.garment_id, asset_sha256=file_sha256(garment),
            topology_sha256=hashlib.sha256(topology.tobytes()).hexdigest(),
            scene_physics_sha256=hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest(),
            source_kind="smoke_initial_state",
            source_sha256=hashlib.sha256(initial.particle_q.tobytes()).hexdigest(),
            normalization_scale_m=float(np.linalg.norm(np.ptp(initial.particle_q[:, :2], axis=0))),
            particle_q=initial.particle_q,
        )
        distances = factorized_cloth_distance(
            positions[-1:], np.zeros_like(positions[-1:]), reference, symmetry="none",
        )
        runtime.reset()
        reset = capture_states(runtime, 0)[0]
        reset_exact = all(
            np.array_equal(getattr(initial, key), getattr(reset, key))
            for key in initial.__dataclass_fields__
        )
        if not reset_exact:
            raise RuntimeError("runtime reset did not restore its initial state")
        summary = {
            "states": reader.manifest.state_count, "actions": reader.manifest.action_count,
            "particles": len(initial.particle_q), "reset_exact": reset_exact,
            "storage": "render_only_v1", "device": args.device,
            "final_distance_from_initial": float(distances.combined_distance[0]),
            "task_success_evaluated": False,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
