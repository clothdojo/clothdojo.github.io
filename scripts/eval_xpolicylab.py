"""Evaluate an XPolicyLab policy server on one ClothDojo garment episode."""

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="XPolicyLab policy directory name")
    parser.add_argument("--model-id", required=True, help="checkpoint or run identifier recorded with results")
    parser.add_argument("--server", default="ws://127.0.0.1:6000")
    parser.add_argument("--xpl-root", type=Path, default=Path("XPolicyLab"))
    parser.add_argument("--task", choices=("shirt_flatten", "shirt_fold"), default="shirt_flatten")
    parser.add_argument("--cloth-visual", choices=("nocs", "white", "rgb"), default="nocs")
    parser.add_argument("--isaac-gpu", help="Isaac GPU ordinal for live RGB observations")
    parser.add_argument("--visual-assets", type=Path, help="downloaded Isaac material assets")
    parser.add_argument("--visual-profile", choices=("train", "test_cloth", "test_background", "test_lighting", "test_all"), default="train")
    parser.add_argument("--visual-seed", type=int, default=42)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, default=Path("outputs/xpolicylab-eval"))
    parser.add_argument("--initial-states", type=Path)
    parser.add_argument("--initial-index", type=int, default=0)
    parser.add_argument("--template", type=Path, help="score the saved trajectory after evaluation")
    parser.add_argument("--protocol", type=Path, help="override the task's scoring protocol")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1 or args.seed < 0 or args.initial_index < 0:
        parser.error("steps must be positive; seed and initial index must be non-negative")
    if not args.policy.isidentifier():
        parser.error("policy must be a Python package name")
    if not args.model_id.strip():
        parser.error("model id must be non-empty")
    if args.protocol is not None and args.template is None:
        parser.error("--protocol requires --template")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.cloth_visual == "rgb" and (args.isaac_gpu is None or args.visual_assets is None):
        parser.error("RGB evaluation requires --isaac-gpu and --visual-assets")
    if args.visual_seed < 0:
        parser.error("visual seed must be non-negative")
    xpl_root = args.xpl_root.expanduser().resolve()
    if not (xpl_root / "XPolicyLab.py").is_file():
        parser.error(f"XPolicyLab source checkout not found: {xpl_root}")
    extension_root = Path(__file__).resolve().parents[1] / "integrations/xpolicylab"
    external_policy = (extension_root / args.policy / "deploy.py").is_file()
    if not (xpl_root / "policy" / args.policy / "deploy.py").is_file() and not external_policy:
        parser.error(f"XPolicyLab policy deployment not found: {args.policy}")
    if args.initial_states is not None and not args.initial_states.is_file():
        parser.error(f"initial states not found: {args.initial_states}")

    import numpy as np
    from clothdojo.assets.bundle import verify_simulation_assets
    from clothdojo.evaluation.scoring import score_trajectory
    from clothdojo.rollout.initial_states import InitialStatePool, validate_pool_compatibility
    from clothdojo.rollout.newton import NewtonTrajectoryEpisode
    from clothdojo.rollout.render import ShirtFlattenCameraObservation
    from clothdojo.rollout.scenario import RolloutScenario
    from clothdojo.rollout.xpolicylab import ClothDojoTaskEnv, run_xpolicylab_deploy
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene
    from clothdojo.trajectory import TrajectoryReader

    # XPolicyLab imports both its top-level transport package and its policy modules.
    sys.path.insert(0, str(xpl_root.parent))
    sys.path.insert(0, str(xpl_root))
    if external_policy:
        import XPolicyLab.policy as policy_package

        policy_package.__path__.append(str(extension_root))
    from client_server.ws.model_client import WsModelClient

    verify_simulation_assets(args.assets)
    pool = InitialStatePool.load(args.initial_states) if args.initial_states else None
    initialization = {}
    if pool is not None:
        if args.initial_index >= pool.state_count:
            parser.error("initial index exceeds the pool")
        spec = ShirtFlattenSpec(garment_id=pool.metadata["garment_id"])
        initialization = {
            "initial_state_index": args.initial_index,
            "initial_state_seed": int(pool.seeds[args.initial_index]),
            "initial_states_sha256": hashlib.sha256(args.initial_states.read_bytes()).hexdigest(),
        }
    else:
        spec = ShirtFlattenSpec()
    prompt = "flatten the shirt" if args.task == "shirt_flatten" else "Fold the shirt."
    scenario = RolloutScenario("xpolicylab", args.task, spec.garment_id,
                               args.seed, prompt, args.steps, 50)
    if args.cloth_visual == "rgb":
        from clothdojo.rollout.isaac import IsaacCameraObservation, isaac_application

        application = isaac_application(args.isaac_gpu)
    else:
        application = nullcontext(None)
    with application as app:
        scene = build_scene(args.assets, spec=spec, device=args.device, visuals=True,
                            cloth_visual="white" if args.cloth_visual == "rgb" else args.cloth_visual)
        runtime = ShirtFlattenRuntime(scene)
        camera = None
        try:
            if pool is not None:
                garment = (args.assets / "garments/converted" / spec.garment_id /
                           f"{spec.garment_id}.usdc")
                validate_pool_compatibility(pool, garment_path=garment,
                                            particle_count=scene.model.particle_count)
                if pool.metadata.get("garment_scale") != spec.garment_scale:
                    raise ValueError("initial-state garment scale differs from the simulation")
                q, qd, _ = pool.select(np.asarray([args.initial_index]))
                runtime.restore_particles(q, qd)
            if args.cloth_visual == "rgb":
                garment = (args.assets / "garments/converted" / spec.garment_id /
                           f"{spec.garment_id}.usdc")
                camera = IsaacCameraObservation(
                    app, runtime, asset_path=garment, assets_root=args.assets,
                    visual_assets=args.visual_assets, visual_profile=args.visual_profile,
                    visual_seed=args.visual_seed,
                )
                camera.begin_episodes((f"{args.task}:{spec.garment_id}:{args.initial_index}:{args.visual_seed}",))
            else:
                camera = ShirtFlattenCameraObservation(scene)
            with NewtonTrajectoryEpisode(
                runtime, (args.output,), (scenario,), assets_root=args.assets,
                observation_provider=camera,
                trajectory_metadata=({"policy": {"system": "XPolicyLab", "name": args.policy,
                                                 "model_id": args.model_id},
                                      "simulation_spec": asdict(spec), **initialization},),
                producer="xpolicylab_rollout", training_eligible=False,
                storage_kind="render_only_v1",
            ) as episode:
                task_env = ClothDojoTaskEnv(episode, scenario)
                client = WsModelClient(url=args.server, evaluation_id="ClothDojo",
                                       trial_id=f"{args.task}-{args.seed}-{args.initial_index}")
                try:
                    report = run_xpolicylab_deploy(task_env, client, policy_name=args.policy)
                finally:
                    client.close()
            reader = TrajectoryReader(args.output)
            reader.validate_render_data()
            score = (score_trajectory(args.output, args.template, assets=args.assets,
                                      protocol_path=args.protocol)
                     if args.template is not None else None)
            summary = {**report, "model_id": args.model_id, "task": args.task, "seed": args.seed,
                       "states": reader.manifest.state_count,
                       "actions": reader.manifest.action_count,
                       "task_success_evaluated": score is not None and score["success"] is not None}
            if score is not None:
                summary["score"] = score
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary, indent=2))
        finally:
            if camera is not None and hasattr(camera, "close"):
                camera.close()
            runtime.close()


if __name__ == "__main__":
    main()
