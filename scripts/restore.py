"""Cold-restore a full state and continue with a bounded segment of saved controls."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() or args.frame < 0 or args.steps < 1:
        parser.error("use a fresh output, non-negative frame and positive step count")

    import numpy as np
    import warp as wp

    from clothdojo.assets.bundle import verify_simulation_assets
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene
    from clothdojo.trajectory import RUNTIME_ACTION_SEMANTICS, TrajectoryReader, TrajectoryWriter
    from clothdojo.trajectory.newton import capture_canonical_actions, capture_states

    verify_simulation_assets(args.assets)
    reader = TrajectoryReader(args.trajectory)
    metadata = dict(reader.manifest.metadata)
    if metadata.get("action_semantics") != RUNTIME_ACTION_SEMANTICS:
        parser.error("recording must contain runtime-applied controls")
    if "simulation_spec" not in metadata:
        parser.error("recording must include simulation_spec")
    if args.frame + args.steps > reader.manifest.action_count:
        parser.error("the requested continuation exceeds the recorded controls")
    frame = reader.state_at(args.frame)  # Reject render-only input; never invent velocities.
    metadata.pop("storage", None)  # The output is a new, self-contained full recording.
    spec = ShirtFlattenSpec(**metadata["simulation_spec"])
    if spec.world_count != 1 or spec.garment_id != metadata.get("asset_id"):
        parser.error("restoration requires one world and a matching garment")
    garment = args.assets / "garments/converted" / spec.garment_id / f"{spec.garment_id}.usdc"
    if hashlib.sha256(garment.read_bytes()).hexdigest() != reader.manifest.scene.asset_fingerprints.get("garment_usdc"):
        parser.error("garment differs from the recording")
    scene = build_scene(args.assets, spec=spec, device=args.device,
                        robot_id=reader.manifest.scene.robot_id)
    runtime = ShirtFlattenRuntime(scene)
    try:
        runtime.restore_state(frame)
        actions = reader.action_arrays()
        origin = {"source_manifest_sha256": hashlib.sha256(
            (args.trajectory / "manifest.json").read_bytes()).hexdigest(),
            "source_frame": args.frame, "kind": "cold_restore_saved_controls"}
        with TrajectoryWriter(
            args.output, control_fps=spec.control_fps, sim_substeps=spec.sim_substeps,
            max_actions=args.steps, scene=reader.manifest.scene,
            metadata={**metadata, "producer": "restore_example", "training_eligible": False,
                      "restoration": origin},
        ) as writer:
            writer.write_initial_state(capture_states(runtime, 0)[0])
            for step in range(args.steps):
                index = args.frame + step
                arms = wp.array(actions["joint_target"][index:index + 1], dtype=wp.float32, device=args.device)
                grippers = wp.array(actions["gripper_open_command"][index:index + 1], dtype=wp.float32, device=args.device)
                runtime.step(arms, grippers)
                writer.append_transition(capture_canonical_actions(runtime, step)[0],
                                         capture_states(runtime, step + 1)[0])
        restored = TrajectoryReader(args.output)
        restored.validate_render_data()
        expected = reader.state_at(args.frame + args.steps)
        actual = restored.state_at(args.steps)
        report = {**origin, "steps": args.steps,
                  "particle_position_max_abs_difference_m": float(np.max(np.abs(
                      actual.particle_q - expected.particle_q))),
                  "note": "Cold restoration clears solver warm starts; continuation is not bitwise replay."}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
