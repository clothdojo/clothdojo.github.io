"""Autonomous flatten-then-fold evaluation using existing Newton, writer and scorers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np


def evaluate(args, isaac_app=None):
    import torch

    from clothdojo.assets import GarmentLibrary
    from clothdojo.evaluation.metric import load_canonical_template
    from clothdojo.evaluation.protocol import load_evaluation_protocol
    from clothdojo.evaluation.scoring import score_evaluation_trajectory
    from clothdojo.evaluation.templates import resolve_canonical_templates
    from clothdojo.rendering import POLICY_RENDER_SIZE, policy_observation_settings
    from clothdojo.rollout.diffusion import AdjacentObservationEpisode, DiffusionPolicy
    from clothdojo.rollout.engine import run_episode
    from clothdojo.rollout.initial_states import (
        InitialStatePool,
        load_initial_state_collection,
        validate_pool_compatibility,
    )
    from clothdojo.rollout.newton import NewtonTrajectoryEpisode
    from clothdojo.rollout.render import ShirtFlattenCameraObservation
    from clothdojo.rollout.scenario import load_rollout_scenario
    from clothdojo.rollout.scheduling import select_assets
    from clothdojo.rollout.trajectory_collection import atomic_write_json, policy_seed_for_batch
    from clothdojo.sim.runtime import ShirtFlattenRuntime
    from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

    torch.set_num_threads(1)
    project = Path(args.project).resolve()
    assets_root = project / "assets"
    states = project / "data/benchmarks/balanced200_v2_20260907/states"
    templates = project / "data/benchmarks/large_support_v1_20260908/templates"
    meta = load_initial_state_collection(states)
    selected = select_assets(
        meta["assets"], seen_assets=args.seen_assets, unseen_assets=args.unseen_assets
    )
    selected = selected[args.shard_index :: args.shard_count]
    flatten_protocol = load_evaluation_protocol(project / "configs/evaluation/shirt_flatten.yaml")
    resolved_templates = resolve_canonical_templates(
        templates,
        [item["asset_id"] for item in selected],
        target_settings=flatten_protocol.target.to_json_dict(),
    )
    policy = DiffusionPolicy(args.checkpoint, device="cuda:0")
    print(json.dumps({"phase": "policy_loaded", "step": policy.identity["step"]}), flush=True)
    scenario_path = (
        "configs/rollout/shirt_flatten.yaml"
        if args.task == "flatten"
        else "configs/rollout/shirt_flatten_fold_eval.yaml"
    )
    base = replace(
        load_rollout_scenario(project / scenario_path),
        action_horizon=policy.cfg["n_action_steps"],
        steps=args.steps,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectories").mkdir(exist_ok=True)
    plan = {
        "policy": policy.identity,
        "scenario": base.to_json_dict(),
        "assets": selected,
        "limit_per_asset": args.limit_per_asset,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "states": str(states),
        "storage_kind": "render_only_v1",
        "observation": policy_observation_settings(
            cloth_visual=args.cloth_visual, visual_profile=args.visual_profile,
            visual_seed=args.visual_seed,
        ),
        "template_sha256": {asset: value[1] for asset, value in resolved_templates.items()},
    }
    plan_path = output / f"plan_{args.shard_index}.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("evaluation output already belongs to a different plan")
    atomic_write_json(plan_path, plan)
    protocols = (
        {
            name: replace(
                load_evaluation_protocol(project / f"configs/evaluation/shirt_{name}.yaml"),
                horizon_seconds=args.steps / 50,
            )
            for name in (("flatten",) if args.task == "flatten" else ("flatten", "fold"))
        }
        if args.steps >= 150
        else {}
    )
    atomic_write_json(
        output / f"protocols_{args.shard_index}.json",
        {name: protocol.to_json_dict() for name, protocol in protocols.items()},
    )
    results = []
    for item in selected:
        asset_id = item["asset_id"]
        print(json.dumps({"phase": "building_scene", "asset": asset_id}), flush=True)
        pool_path = states / asset_id / "states.npz"
        pool = InitialStatePool.load(pool_path)
        count = min(args.limit_per_asset, pool.state_count)
        garment = GarmentLibrary(assets_root).converted(asset_id)
        validate_pool_compatibility(pool, garment_path=garment, particle_count=pool.particle_count)
        if "cold_robot_state" in pool.metadata:
            raise ValueError("this evaluation requires the canonical hanging-cloth initial pool")
        with pool_path.open("rb") as stream:
            pool_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if pool_hash != item["sha256"]:
            raise ValueError(f"initial state pool changed: {asset_id}")
        scene = build_scene(
            assets_root,
            task_name=base.task_name,
            spec=ShirtFlattenSpec(garment_id=asset_id, world_count=count),
            device="cuda:0",
            visuals=args.cloth_visual != "rgb",
            cloth_visual="white" if args.cloth_visual == "rgb" else "nocs",
        )
        runtime = ShirtFlattenRuntime(scene)
        print(json.dumps({"phase": "initializing_camera", "asset": asset_id}), flush=True)
        if args.cloth_visual == "rgb":
            from clothdojo.rollout.isaac import IsaacCameraObservation

            camera = IsaacCameraObservation(
                isaac_app, runtime, asset_path=garment, assets_root=assets_root,
                visual_profile=args.visual_profile, visual_seed=args.visual_seed,
            )
        else:
            camera = ShirtFlattenCameraObservation(
                scene, width=POLICY_RENDER_SIZE[0], height=POLICY_RENDER_SIZE[1]
            )
        runtime.reset()
        runtime.restore_particles(
            np.ascontiguousarray(pool.particle_q[:count]),
            np.ascontiguousarray(pool.particle_qd[:count]),
        )
        scenarios = tuple(
            replace(base, garment_id=asset_id, seed=int(pool.seeds[i])) for i in range(count)
        )
        targets = tuple(output / "trajectories" / f"{asset_id}_{i:03d}" for i in range(count))
        print(json.dumps({"phase": "initializing_appearance", "asset": asset_id}), flush=True)
        appearances = (
            camera.begin_episodes(tuple(path.name for path in targets))
            if args.cloth_visual == "rgb" else None
        )
        seed = policy_seed_for_batch(np.asarray(pool.seeds[:count], dtype=np.int32))
        if any(path.exists() for path in targets):
            raise FileExistsError(
                "use a fresh output for evaluation; partial trajectories are never reused"
            )
        metadata = tuple(
            {
                "asset_id": asset_id,
                "policy_kind": "lerobot_diffusion",
                "policy": policy.identity,
                "policy_seed": seed,
                "replicate": 0,
                "cloth_visual": args.cloth_visual,
                "observation": plan["observation"],
                **({"visual_layout": appearances[i]} if appearances is not None else {}),
                "initial_state": {"index": i, "seed": int(pool.seeds[i]), "pool_sha256": pool_hash},
            }
            for i in range(count)
        )
        with NewtonTrajectoryEpisode(
            runtime,
            targets,
            scenarios,
            assets_root=assets_root,
            observation_provider=camera,
            trajectory_metadata=metadata,
            producer="clothdojo_diffusion_eval",
            training_eligible=False,
            storage_kind="render_only_v1",
        ) as episode:
            print(json.dumps({"phase": "running", "asset": asset_id, "steps": args.steps}), flush=True)
            report = run_episode(
                AdjacentObservationEpisode(episode, policy.cfg["n_action_steps"]),
                policy,
                scenarios[0],
                mode="chunk",
                exec_horizon=policy.cfg["n_action_steps"],
                policy_seed=seed,
            )
        print(json.dumps({"phase": "scoring", "asset": asset_id}), flush=True)
        for index, path in enumerate(targets):
            row = {
                "asset_id": asset_id,
                "initial_state_index": index,
                "path": str(path),
                "split": item.get("split"),
                "execution": report.to_json_dict(),
            }
            for name, protocol in protocols.items():
                scored = score_evaluation_trajectory(
                    path,
                    load_canonical_template(resolved_templates[asset_id][0]),
                    protocol,
                    assets_root=assets_root,
                )
                row[name] = scored.result.to_json_dict()
            results.append(row)
        atomic_write_json(output / f"results_{args.shard_index}.json", results)
        print(json.dumps({"asset": asset_id, "completed_cases": len(results)}), flush=True)
        if args.cloth_visual == "rgb":
            camera.close()
        del episode, camera, runtime, scene
        torch.cuda.empty_cache()
    atomic_write_json(
        output / f"status_{args.shard_index}.json", {"status": "complete", "cases": len(results)}
    )


def main():
    from clothdojo.assets.visual_benchmark import VISUAL_PROFILES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", default=".")
    parser.add_argument("--seen-assets", type=int, default=12)
    parser.add_argument("--unseen-assets", type=int, default=12)
    parser.add_argument("--limit-per-asset", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--task", choices=("flatten", "flatten_fold"), default="flatten_fold")
    parser.add_argument("--cloth-visual", choices=("nocs", "rgb"), default="nocs")
    parser.add_argument("--visual-profile", choices=VISUAL_PROFILES)
    parser.add_argument("--visual-seed", type=int, default=0)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count or args.limit_per_asset < 1:
        parser.error("invalid shard or case count")
    from clothdojo.rendering import policy_observation_settings

    policy_observation_settings(
        cloth_visual=args.cloth_visual, visual_profile=args.visual_profile,
        visual_seed=args.visual_seed,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    application = nullcontext(None)
    if args.cloth_visual == "rgb":
        from clothdojo.rollout.isaac import isaac_application

        application = isaac_application(args.gpu)
    with application as app:
        try:
            evaluate(args, app)
        except BaseException as error:
            # Kit may terminate Python during close(), before an uncaught
            # exception reaches the interpreter. Persist the failure first.
            import traceback
            from clothdojo.rollout.trajectory_collection import atomic_write_json

            traceback.print_exc()
            output = Path(args.output).resolve()
            output.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output / f"status_{args.shard_index}.json", {
                "status": "failed", "error": str(error), "traceback": traceback.format_exc(),
            })
            raise


if __name__ == "__main__":
    main()
