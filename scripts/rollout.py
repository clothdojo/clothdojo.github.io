"""Run a bundled Pi0.5 policy and save its trajectory and camera views."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rollout"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("chunk", "rtc"), default="chunk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-state", action="store_true",
                        help="retain dynamics/grasp state for restoration and mirroring")
    parser.add_argument("--initial-states", type=Path)
    parser.add_argument("--initial-index", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0 or args.initial_index < 0:
        parser.error("steps must be positive; seed and initial index must be non-negative")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    import cv2
    import numpy as np

    from clothdojo.assets.bundle import verify_simulation_assets
    from clothdojo.rollout.engine import run_episode
    from clothdojo.rollout.initial_states import InitialStatePool, validate_pool_compatibility
    from clothdojo.rollout.newton import NewtonTrajectoryEpisode
    from clothdojo.rollout.policy import Pi05Policy
    from clothdojo.rollout.render import ShirtFlattenCameraObservation
    from clothdojo.rollout.scenario import RolloutScenario
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene
    from clothdojo.trajectory import TrajectoryReader

    verify_simulation_assets(args.assets)
    policy = Pi05Policy(args.model, device=args.device)
    task = policy.metadata["task"]
    prompt = "flatten the shirt" if task == "shirt_flatten" else "Fold the shirt."
    initial_path = args.initial_states or args.model / "initial_states.npz"
    pool = None
    initialization = {}
    if initial_path.is_file():
        pool = InitialStatePool.load(initial_path)
        if args.initial_index >= pool.state_count:
            parser.error("initial index exceeds the pool")
        spec = ShirtFlattenSpec(garment_id=pool.metadata["garment_id"])
        initialization = {"initial_state_index": args.initial_index,
                          "initial_state_seed": int(pool.seeds[args.initial_index]),
                          "initial_states_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest()}
    elif args.initial_states is not None:
        parser.error(f"initial states not found: {initial_path}")
    else:
        spec = ShirtFlattenSpec()
    scenario = RolloutScenario("example", task, spec.garment_id, args.seed,
                               prompt, args.steps, policy.action_horizon)
    scene = build_scene(args.assets, spec=spec, device=args.device, visuals=True,
                        cloth_visual=policy.metadata["cloth_visual"])
    runtime = ShirtFlattenRuntime(scene)
    try:
        if pool is not None:
            garment = args.assets / "garments/converted" / spec.garment_id / f"{spec.garment_id}.usdc"
            validate_pool_compatibility(pool, garment_path=garment,
                                        particle_count=scene.model.particle_count)
            if pool.metadata.get("garment_scale") != spec.garment_scale:
                raise ValueError("initial-state garment scale differs from the simulation")
            q, qd, _ = pool.select(np.asarray([args.initial_index]))
            runtime.restore_particles(q, qd)
        camera = ShirtFlattenCameraObservation(scene)
        with NewtonTrajectoryEpisode(
            runtime, (args.output,), (scenario,), assets_root=args.assets,
            observation_provider=camera,
            trajectory_metadata=({"policy": policy.metadata, "simulation_spec": asdict(spec),
                                  **initialization},),
            producer="pi05_rollout", training_eligible=False,
            storage_kind="full" if args.full_state else "render_only_v1",
        ) as episode:
            first = episode.observe()
            report = run_episode(episode, policy, scenario, mode=args.mode,
                                 exec_horizon=10, policy_seed=args.seed)
            last = episode.observe()
        reader = TrajectoryReader(args.output)
        reader.validate_render_data()
        for label, observation in (("initial", first), ("final", last)):
            panels = [pixels[0] for key, pixels in observation.items()
                      if key.startswith("observation.images.")]
            mosaic = cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(args.output / f"{label}.png"), mosaic):
                raise RuntimeError("could not save camera preview")
        summary = report.to_json_dict()
        summary.update({"states": reader.manifest.state_count,
                        "actions": reader.manifest.action_count,
                        "task_success_evaluated": False})
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
