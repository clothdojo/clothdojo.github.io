"""ClothDojo command-line interface."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path


def _training_config_schema(path: str | Path) -> str:
    import yaml

    source = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read training config {source}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("schema"), str):
        return ""
    return value["schema"]


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args, passthrough = parser.parse_known_args(argv)
    if args.command in {"generate", "replay"}:
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.quickstart import generate, replay

        try:
            if args.command == "generate":
                generate(args.config, Path.cwd())
            else:
                replay(args.source, Path.cwd(), benchmark=args.benchmark, gpu=args.gpu)
        except KeyboardInterrupt:
            parser.exit(130, "Cancelled. Partial output and logs have been kept.\n")
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f"{error}\n")
        return
    if args.command == "env" and args.env_command == "check":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.diagnostics import inspect_environment

        report = inspect_environment(
            include_training=args.training,
            smoke=args.smoke,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["compatible"]:
            raise SystemExit(1)
        return
    if args.command == "steam" and args.steam_command == "merge":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.training.steam import merge_steam_members

        output = merge_steam_members(
            args.member,
            args.output,
            overwrite=args.overwrite,
        )
        print(json.dumps({"checkpoint": str(output)}, indent=2, sort_keys=True))
        return
    if args.command == "train":
        if args.training_kind == "diffusion":
            if any(value is not None for value in
                   (args.gate, args.member_index, args.advantages_config)):
                parser.error("diffusion does not accept OpenPI/STEAM/CFG options")
            from clothdojo.training.diffusion import launch

            launch(args.config, passthrough, check=args.check)
            return
        if args.training_kind == "steam" and args.member_index is None:
            parser.error("STEAM training requires --member-index")
        if args.training_kind != "steam" and args.member_index is not None:
            parser.error("--member-index is only valid for STEAM training")
        if args.advantages_config is not None and args.training_kind != "cfg":
            parser.error("--advantages-config is only valid for CFG training")
        if args.training_kind == "cfg" and args.advantages_config is None:
            parser.error("CFG training requires --advantages-config")
        if args.training_kind == "cfg":
            if args.gate is not None:
                parser.error("--gate is only valid for openpi-jax training")
            from clothdojo.training.cfg_pipeline import (
                inspect_cfg_pipeline,
                run_cfg_pipeline,
            )

            result = (
                inspect_cfg_pipeline(
                    advantage_config=args.advantages_config,
                    cfg_config=args.config,
                    cfg_overrides=passthrough,
                )
                if args.check
                else run_cfg_pipeline(
                    advantage_config=args.advantages_config,
                    cfg_config=args.config,
                    cfg_overrides=passthrough,
                )
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.training_kind == "openpi-jax":
            if _training_config_schema(args.config) == "clothdojo_pi05_jax_matrix":
                if passthrough:
                    parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
                if args.gate is not None:
                    parser.error("an OpenPI/JAX matrix does not accept --gate")
                from clothdojo.training.matrix import (
                    compose_jax_sft_matrix,
                    inspect_jax_sft_matrix,
                    run_jax_sft_matrix,
                )

                matrix = compose_jax_sft_matrix(args.config)
                result = (
                    inspect_jax_sft_matrix(matrix, inspect_artifacts=True)
                    if args.check
                    else run_jax_sft_matrix(matrix)
                )
                print(json.dumps(result, indent=2, sort_keys=True))
                return
            from clothdojo.training.jax_sft import (
                compose_jax_sft_spec,
                inspect_jax_sft,
                run_jax_sft,
            )

            spec = compose_jax_sft_spec(args.config, passthrough)
            if args.check:
                print(json.dumps(inspect_jax_sft(spec), indent=2, sort_keys=True))
                return
            if args.gate is None:
                parser.error("openpi-jax training requires --gate")
            result = run_jax_sft(spec, args.gate)
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.gate is not None:
            parser.error("--gate is only valid for openpi-jax training")
        if args.check:
            from clothdojo.training.rlinf import check as check_rlinf_training

            report = check_rlinf_training(
                kind=args.training_kind,
                config_path=args.config,
                overrides=passthrough,
                steam_member_index=args.member_index,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        from clothdojo.training.rlinf import run as run_rlinf_training

        run_rlinf_training(
            kind=args.training_kind,
            config_path=args.config,
            overrides=passthrough,
            steam_member_index=args.member_index,
        )
        return
    if args.command == "policy" and args.policy_command == "serve-sft":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.rollout.policy import serve_sft_policy

        serve_sft_policy(
            args.checkpoint,
            args.norm_stats,
            num_steps=args.num_steps,
            image_preprocessing=args.image_preprocessing,
            rtc_guidance_horizon=args.rtc_guidance_horizon,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            rtc_prefix_schedule=args.rtc_prefix_schedule,
            device=args.device,
            host=args.host,
            port=args.port,
        )
        return
    if args.command == "policy" and args.policy_command == "serve-jax":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.rollout.policy import serve_jax_policy

        serve_jax_policy(
            args.checkpoint,
            args.norm_stats,
            args.training_config,
            num_steps=args.num_steps,
            image_preprocessing=args.image_preprocessing,
            host=args.host,
            port=args.port,
        )
        return
    if args.command == "policy" and args.policy_command == "serve-groot":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.rollout.groot import serve_groot_policy

        serve_groot_policy(
            args.checkpoint,
            args.norm_stats,
            groot_root=args.groot_root,
            checkpoint_sha256=args.checkpoint_sha256,
            embodiment_tag=args.embodiment_tag,
            device=args.device,
            host=args.host,
            port=args.port,
        )
        return
    if args.command == "policy" and args.policy_command == "serve-diffusion":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.rollout.diffusion import serve_diffusion_policy

        serve_diffusion_policy(
            args.checkpoint,
            args.norm_stats,
            image_preprocessing=args.image_preprocessing,
            device=args.device,
            host=args.host,
            port=args.port,
        )
        return
    if args.command == "policy" and args.policy_command == "serve-starvla":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.starvla_policy_server import serve_starvla_policy

        serve_starvla_policy(
            args.checkpoint,
            starvla_root=args.starvla_root,
            device=args.device,
            host=args.host,
            port=args.port,
        )
        return
    if args.command == "policy" and args.policy_command == "convert-jax":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.rollout.conversion import convert_jax_checkpoint

        result = convert_jax_checkpoint(
            args.checkpoint,
            args.output,
            args.norm_stats,
            args.parity_config,
            parity_batch_size=args.parity_batch_size,
            parity_execution_horizon=args.parity_execution_horizon,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "policy" and args.policy_command == "serve-cfg":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.rollout.policy import serve_cfg_policy

        serve_cfg_policy(
            args.checkpoint,
            args.norm_stats,
            num_steps=args.num_steps,
            image_preprocessing=args.image_preprocessing,
            guidance_scale=args.guidance_scale,
            guidance_type=args.guidance_type,
            positive_only_conditional=args.positive_only_conditional,
            autocast=not args.no_autocast,
            rtc_guidance_horizon=args.rtc_guidance_horizon,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            rtc_prefix_schedule=args.rtc_prefix_schedule,
            device=args.device,
            host=args.host,
            port=args.port,
        )
        return
    if args.command == "data" and args.data_command == "prepare-isaac-assets":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.isaac_render import prepare_isaac_assets

        result = prepare_isaac_assets(args.output, include_room=args.include_room)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "augment-mirror":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.mirror import augment_mirror

        result = augment_mirror(args.source, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "trim-home-returns":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.home_trim import trim_home_returns

        result = trim_home_returns(
            args.source,
            args.output,
            tolerance=args.tolerance,
            min_hold_frames=args.min_hold_frames,
            min_output_actions=args.min_output_actions,
            workers=args.workers,
            plan_only=args.plan_only,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "canonicalize-actions":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.action_canonicalization import canonicalize_action_collection

        result = canonicalize_action_collection(
            args.source,
            args.output,
            plan_only=args.plan_only,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "prepare-recap":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.recap import prepare_recap_plan

        result = prepare_recap_plan(
            args.expert,
            args.hil_rollout,
            args.corrections,
            output=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "prepare-recap-mirrored":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.recap import prepare_recap_mirrored_plan

        result = prepare_recap_mirrored_plan(
            args.plan,
            args.expert,
            args.hil_rollout,
            args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "prepare-recap-value-split":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.recap import prepare_recap_value_split

        result = prepare_recap_value_split(
            args.plan,
            args.output,
            eval_seen_assets=args.eval_seen_assets,
            eval_unseen_assets=args.eval_unseen_assets,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "prepare-recap-value-trajectory-split":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.recap import prepare_recap_value_trajectory_split

        result = prepare_recap_value_trajectory_split(
            args.plan,
            args.output,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "compute-recap-returns":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.recap import compute_recap_returns

        result = compute_recap_returns(
            args.plan,
            args.dataset_root,
            tag=args.tag,
            c_fail_coef=args.c_fail_coef,
            gamma=args.gamma,
            failure_horizon_steps=args.failure_horizon_steps,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "reuse-recap-export":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.recap import reuse_recap_export

        result = reuse_recap_export(args.plan, args.dataset_root, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "export-recap":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        if (args.shard_index is None) != (args.shard_count is None):
            parser.error("--shard-index and --shard-count must be provided together")
        from clothdojo.data.recap import export_recap_dataset, inspect_recap_export

        if args.plan_only:
            result = inspect_recap_export(
                args.plan,
                args.dataset,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                pair_member=args.pair_member,
            )
        else:
            if args.output is None:
                parser.error("RECAP export requires --output unless --plan-only is used")
            result = export_recap_dataset(
                args.plan,
                args.dataset,
                args.output,
                assets_root=args.assets,
                device=args.device,
                repo_id=args.repo_id,
                image_writer_threads=args.image_writer_threads,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                cloth_visual=args.cloth_visual,
                pair_member=args.pair_member,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "merge-lerobot-shards":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.lerobot import merge_lerobot_shards

        result = merge_lerobot_shards(args.shard, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "retarget":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.retarget import retarget

        result = retarget(
            args.source,
            args.output,
            target_robot=args.target_robot,
            assets=args.assets,
            device=args.device,
            **({"devices": tuple(args.devices)} if args.devices is not None else {}),
            **(
                {"workers_per_device": args.workers_per_device}
                if args.workers_per_device != 1
                else {}
            ),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "export-lerobot":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        if args.include_room and args.isaac_seed is None:
            parser.error("--include-room requires --isaac-seed")
        if args.randomize is not None and args.isaac_seed is None:
            parser.error("--randomize requires --isaac-seed")
        if args.visual_profile is not None and args.isaac_seed is None:
            parser.error("--visual-profile requires --isaac-seed")
        if args.visual_profile is not None and args.include_room:
            parser.error("--visual-profile selects backgrounds; omit --include-room")
        if args.isaac_seed is not None and args.cloth_visual != "nocs":
            parser.error("--cloth-visual is only supported by Newton export")
        if (args.shard_index is None) != (args.shard_count is None):
            parser.error("--shard-index and --shard-count must be provided together")
        source_root = Path(args.source).expanduser()
        source_manifest = source_root / "meta.json"
        if not source_manifest.is_file():
            source_manifest = source_root / "manifest.json"
        source_schema = None
        if source_manifest.is_file():
            try:
                source_schema = json.loads(source_manifest.read_text(encoding="utf-8")).get(
                    "schema"
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot read source manifest: {source_manifest}: {exc}"
                ) from exc
        if source_schema in {
            "clothdojo_rollout",
            "clothdojo_evaluation_condition",
            "clothdojo_retarget_collection",
        } or args.isaac_seed is not None:
            if args.plan_only:
                if args.isaac_seed is not None:
                    from clothdojo.data.isaac_render import (
                        inspect_isaac_rollout_export,
                        parse_visual_randomization,
                    )

                    result = inspect_isaac_rollout_export(
                        args.source,
                        seed=args.isaac_seed,
                        include_room=args.include_room,
                        visual_profile=args.visual_profile,
                        min_segment_frames=args.min_segment_frames,
                        whole_trajectory_good_only=args.whole_trajectory_good_only,
                        shard_index=args.shard_index,
                        shard_count=args.shard_count,
                        randomize=(
                            args.randomize
                            if args.randomize is not None
                            else parse_visual_randomization("all")
                        ),
                    )
                else:
                    from clothdojo.data.rollout import inspect_rollout_collection_export

                    result = inspect_rollout_collection_export(
                        args.source,
                        shard_index=args.shard_index,
                        shard_count=args.shard_count,
                        cloth_visual=args.cloth_visual,
                    )
                print(json.dumps(result, indent=2, sort_keys=True))
                return
            output = args.output or str(source_root / "lerobot")
            if args.isaac_seed is not None:
                from clothdojo.data.isaac_render import (
                    export_isaac_rollout_collection,
                    parse_visual_randomization,
                )

                result = export_isaac_rollout_collection(
                    args.source,
                    output,
                    assets_root=args.assets,
                    visual_assets_root=args.visual_assets,
                    device=args.device,
                    seed=args.isaac_seed,
                    include_room=args.include_room,
                    visual_profile=args.visual_profile,
                    min_segment_frames=args.min_segment_frames,
                    whole_trajectory_good_only=args.whole_trajectory_good_only,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                    randomize=(
                        args.randomize
                        if args.randomize is not None
                        else parse_visual_randomization("all")
                    ),
                    repo_id=args.repo_id,
                    image_writer_threads=args.image_writer_threads,
                    _exit_process=True,
                )
            else:
                from clothdojo.data.rollout import export_rollout_collection

                result = export_rollout_collection(
                    args.source,
                    output,
                    assets_root=args.assets,
                    device=args.device,
                    repo_id=args.repo_id,
                    image_writer_threads=args.image_writer_threads,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                    cloth_visual=args.cloth_visual,
                    **({"resume": True} if args.resume else {}),
                )
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.isaac_seed is not None:
            parser.error("--isaac-seed is only supported for canonical rollout data")
        from clothdojo.data.rollout import (
            export_trajectory_dataset,
            inspect_trajectory_dataset_export,
        )

        if args.plan_only:
            result = inspect_trajectory_dataset_export(
                args.source,
                min_segment_frames=args.min_segment_frames,
                whole_trajectory_good_only=args.whole_trajectory_good_only,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                cloth_visual=args.cloth_visual,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        output = args.output or str(source_root / "lerobot")
        result = export_trajectory_dataset(
            args.source,
            output,
            assets_root=args.assets,
            device=args.device,
            repo_id=args.repo_id,
            image_writer_threads=args.image_writer_threads,
            min_segment_frames=args.min_segment_frames,
            whole_trajectory_good_only=args.whole_trajectory_good_only,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            cloth_visual=args.cloth_visual,
            **({"resume": True} if args.resume else {}),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "render-review":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.rollout_review import render_review

        review_options = {
            "assets_root": args.assets,
            "device": args.device,
            "fps": args.fps,
            "camera_name": args.camera,
            "output": args.output,
        }
        if args.trajectory is not None:
            review_options["trajectory_ids"] = args.trajectory
        result = render_review(args.source, **review_options)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "score-fold-quality":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.fold_quality_collection import score_fold_quality_collection

        result = score_fold_quality_collection(
            args.source, args.output, assets_root=args.assets,
            reference=args.reference,
            sample_stride=args.sample_stride, workers=args.workers,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "compute-norm-stats":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.training.openpi import compute_norm_stats

        result = compute_norm_stats(
            args.dataset,
            args.output,
            block_size=args.block_size,
            action_chunk_boundary=args.action_chunk_boundary,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "collect" and args.collect_command == "serve":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.collection.web import serve

        serve(args.config)
        return
    if args.command == "benchmark" and args.benchmark_command == "check-protocol":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.protocol import load_evaluation_protocol

        protocol = load_evaluation_protocol(args.protocol)
        result = protocol.to_json_dict()
        result.update(
            {
                "fingerprint": protocol.fingerprint,
                "horizon_steps": protocol.horizon_steps,
                "metric_stride": protocol.metric_stride,
            }
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "benchmark" and args.benchmark_command == "run":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.managed import run_managed_evaluation

        result = run_managed_evaluation(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "benchmark" and args.benchmark_command == "make-templates":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.templates import generate_template_collection

        result = generate_template_collection(
            args.states,
            args.protocol,
            args.output,
            assets_root=args.assets,
            gpus=_comma_separated(args.gpus, "gpus"),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "benchmark" and args.benchmark_command == "summarize":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.results import (
            FORMAL_BENCHMARK_PANEL_ID,
            summarize_paired_conditions,
            validate_formal_evaluation_summary,
        )

        result = summarize_paired_conditions(args.condition)
        if result.get("benchmark", {}).get("panel_id") == FORMAL_BENCHMARK_PANEL_ID:
            validate_formal_evaluation_summary(result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "benchmark" and args.benchmark_command == "score":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.collection import score_evaluation_condition

        result = score_evaluation_condition(
            args.condition,
            args.protocol,
            args.templates,
            workers=args.workers,
            rescore_from=args.rescore_from,
            rescore_templates_from=args.rescore_templates_from,
            output=args.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if passthrough:
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
    if args.command == "rollout" and args.rollout_command == "check-states":
        from clothdojo.rollout.state_quality import check_initial_state_collection

        report = check_initial_state_collection(
            args.states,
            assets_root=args.assets,
            report_path=args.report,
            repair_output=args.repair_output,
            gpus=_comma_separated(args.gpus, "gpus"),
            review_rollout=args.review_rollout,
            workers=args.workers,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "rollout" and args.rollout_command == "make-states":
        from clothdojo.rollout.initial_states import generate_initial_state_collection

        report = generate_initial_state_collection(
            args.assets,
            args.output,
            expert_trajectories=args.expert_trajectories,
            seen_assets=args.seen_assets,
            unseen_assets=args.unseen_assets,
            states_per_asset=args.states_per_asset,
            gpus=_comma_separated(args.gpus, "gpus"),
            candidate_batch=args.batch_size,
            max_candidates=args.max_candidates,
            seed=args.seed,
            drop_seconds=args.drop_seconds,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "rollout" and args.rollout_command == "run":
        from clothdojo.rollout.collection import run_rollout_collection

        success_options = {}
        if args.success_quota is not None:
            success_options = {
                "success_quota": args.success_quota,
                "success_initial_attempts": args.success_initial_attempts,
                "success_protocol_path": args.success_protocol,
                "success_templates_path": args.success_templates,
            }
        shard_options = {}
        if args.asset_shard_index is not None or args.asset_shard_count is not None:
            shard_options = {
                "asset_shard_index": args.asset_shard_index,
                "asset_shard_count": args.asset_shard_count,
            }
        prepare_options = {"prepare_only": True} if args.prepare_only else {}
        if args.storage_kind != "full":
            prepare_options["storage_kind"] = args.storage_kind
        report = run_rollout_collection(
            args.scenario,
            args.states,
            _comma_separated(args.policy, "policy"),
            args.output,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            assets_root=args.assets,
            gpus=_comma_separated(args.gpus, "gpus"),
            limit_per_asset=args.limit_per_asset,
            seen_assets=args.seen_assets,
            unseen_assets=args.unseen_assets,
            execution=args.execution,
            exec_horizon=args.exec_horizon,
            rtc_delay_steps=args.rtc_delay_steps,
            batch_size=args.batch_size,
            **prepare_options,
            **shard_options,
            **success_options,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "assets":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.assets.workstations import (
            inspect_workstation_assets,
            prepare_workstation_assets,
        )

        if args.assets_command == "prepare-workstations":
            report = prepare_workstation_assets(args.source, args.output)
        elif args.assets_command == "check-workstations":
            report = inspect_workstation_assets(args.assets)
        else:
            parser.error("an assets command is required")
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "sim" and args.sim_command == "render-workstations":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.robots import robot_ids
        from clothdojo.sim.workstation_review import render_workstation_review

        selected = robot_ids() if args.robot == "all" else (args.robot,)
        reports = [
            render_workstation_review(
                robot,
                assets_root=args.assets,
                output_dir=args.output,
                device=args.device,
                egl_device=args.egl_device,
                views=tuple(args.views),
            )
            for robot in selected
        ]
        print(json.dumps(reports, indent=2, sort_keys=True))
        return
    if args.command == "sim" and args.sim_command == "check":
        from clothdojo.sim.check import run_shirt_flatten_check

        report = run_shirt_flatten_check(
            args.assets,
            device=args.device,
            world_count=args.worlds,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["compatible"]:
            raise SystemExit(1)
        return
    parser.error("a command is required")


def _parser() -> argparse.ArgumentParser:
    from clothdojo.data.annotations import DEFAULT_MIN_SEGMENT_FRAMES

    parser = argparse.ArgumentParser(prog="clothdojo")
    commands = parser.add_subparsers(dest="command")
    generate = commands.add_parser("generate", help="generate trajectories using a small run config")
    generate.add_argument("--config", default="configs/generation/flatten.yaml",
                          help="run YAML (default: configs/generation/flatten.yaml)")
    replay = commands.add_parser("replay", help="export high and low views of a saved collection")
    replay.add_argument("source", help="trajectory collection printed by generate")
    replay.add_argument("--benchmark", default="benchmark", help="benchmark root containing assets/")
    replay.add_argument("--gpu", type=int, default=0)

    environment = commands.add_parser("env", help="inspect the project environment")
    environment_commands = environment.add_subparsers(dest="env_command")
    check = environment_commands.add_parser("check", help="validate runtime versions")
    check.add_argument(
        "--training",
        action="store_true",
        help="also require the optional RLINF training stack",
    )
    check.add_argument(
        "--smoke",
        action="store_true",
        help="run a deterministic Newton build/collide/step smoke test on CPU",
    )

    simulation = commands.add_parser("sim", help="run Newton scene checks")
    simulation_commands = simulation.add_subparsers(dest="sim_command")
    simulation_check = simulation_commands.add_parser(
        "check", help="validate the migrated shirt-flattening slice"
    )
    simulation_check.add_argument("--assets", default="assets")
    simulation_check.add_argument("--device", default="cuda:0")
    simulation_check.add_argument("--worlds", type=int, default=1)

    from clothdojo.robots import robot_ids

    render_workstations = simulation_commands.add_parser(
        "render-workstations", help="render fixed USD home and attach frames with Newton"
    )
    render_workstations.add_argument("--robot", choices=("all", *robot_ids()), default="all")
    render_workstations.add_argument("--assets", default="assets")
    render_workstations.add_argument("--output", default="outputs/workstation_review")
    render_workstations.add_argument("--device", default="cpu")
    render_workstations.add_argument("--egl-device", type=int, default=0)
    render_workstations.add_argument(
        "--views",
        nargs="+",
        choices=("overall", "left", "right", "left_wrist", "right_wrist"),
        default=("overall", "left", "right"),
        help="overall, attach close-ups, or wrist-camera installation close-ups",
    )

    assets = commands.add_parser("assets", help="deploy fixed workstation asset snapshots")
    asset_commands = assets.add_subparsers(dest="assets_command")
    prepare_workstations = asset_commands.add_parser("prepare-workstations")
    prepare_workstations.add_argument("--source", required=True)
    prepare_workstations.add_argument("--output", default="assets")
    check_workstations = asset_commands.add_parser("check-workstations")
    check_workstations.add_argument("--assets", default="assets")

    rollout = commands.add_parser("rollout", help="generate initial states and policy trajectories")
    rollout_commands = rollout.add_subparsers(dest="rollout_command")
    check_states = rollout_commands.add_parser(
        "check-states",
        help="check self-intersections and optionally regenerate invalid state slots",
    )
    check_states.add_argument("--states", required=True)
    check_states.add_argument("--assets", default="assets")
    check_states.add_argument("--report", required=True)
    check_states.add_argument("--repair-output", help="new collection for repaired states")
    check_states.add_argument(
        "--review-rollout",
        help="sequential review: full-asset skips reject assets; partial skips reject states",
    )
    check_states.add_argument("--gpus", default="0", help="simulation GPU ids used for repair")
    check_states.add_argument(
        "--workers", type=int, default=1, help="parallel CPU workers used by the geometry audit"
    )
    make_states = rollout_commands.add_parser(
        "make-states",
        help="select garments and generate one validated state pool per asset",
    )
    make_states.add_argument("--seen-assets", type=int, default=0)
    make_states.add_argument("--unseen-assets", type=int, required=True)
    make_states.add_argument("--states-per-asset", type=int, default=32)
    make_states.add_argument(
        "--expert-trajectories",
        default=os.environ.get("CLOTHDOJO_EXPERT_TRAJECTORIES"),
        help="expert trajectory or LeRobot dataset used to identify seen assets",
    )
    make_states.add_argument("--output", required=True)
    make_states.add_argument("--assets", default="assets")
    make_states.add_argument("--gpus", default="0", help="comma-separated simulation GPU ids")
    make_states.add_argument("--batch-size", type=int, default=8)
    make_states.add_argument("--max-candidates", type=int, default=512)
    make_states.add_argument("--seed", type=int, default=260815)
    make_states.add_argument("--drop-seconds", type=float, default=3.0)
    run_rollout = rollout_commands.add_parser(
        "run",
        help="run an SFT or CFG policy over an initial-state pool",
    )
    run_rollout.add_argument("--scenario", required=True)
    run_rollout.add_argument("--states", required=True)
    run_rollout.add_argument(
        "--policy",
        default="ws://127.0.0.1:8080",
        help="one endpoint or a comma-separated endpoint pool",
    )
    run_rollout.add_argument("--expected-checkpoint-sha256")
    run_rollout.add_argument("--output", required=True)
    run_rollout.add_argument("--assets", default="assets")
    run_rollout.add_argument("--gpus", default="0", help="comma-separated simulation GPU ids")
    run_rollout.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Newton worlds rendered and inferred together per GPU (default: 4)",
    )
    run_rollout.add_argument("--limit-per-asset", type=int)
    run_rollout.add_argument(
        "--storage-kind", choices=("full", "render_only_v1"), default="full",
        help="full restart state or compact poses/actions for rendering and geometry scoring",
    )
    run_rollout.add_argument(
        "--success-quota",
        type=int,
        help="adaptively retain this many successful trajectories per asset",
    )
    run_rollout.add_argument(
        "--success-initial-attempts",
        type=int,
        default=16,
        help="attempts before deciding whether to use the remaining states (default: 16)",
    )
    run_rollout.add_argument(
        "--success-protocol",
        help="evaluation protocol defining success for quota selection",
    )
    run_rollout.add_argument(
        "--success-templates",
        help="complete canonical-template collection used for quota selection",
    )
    run_rollout.add_argument("--seen-assets", type=int)
    run_rollout.add_argument("--unseen-assets", type=int)
    run_rollout.add_argument("--asset-shard-index", type=int)
    run_rollout.add_argument("--asset-shard-count", type=int)
    run_rollout.add_argument(
        "--prepare-only",
        action="store_true",
        help="initialize and validate rollout metadata without starting workers",
    )
    run_rollout.add_argument(
        "--execution",
        choices=("chunk", "rtc"),
        default="chunk",
        help="independent chunks or fixed-delay guided RTC (default: chunk)",
    )
    run_rollout.add_argument(
        "--exec-horizon",
        type=int,
        default=10,
        help="actions executed before replanning (default: 10)",
    )
    run_rollout.add_argument(
        "--rtc-delay-steps",
        type=int,
        default=0,
        help="logical inference delay used by reproducible RTC rollout (default: 0)",
    )

    policy = commands.add_parser("policy", help="prepare and serve rollout policies")
    policy_commands = policy.add_subparsers(dest="policy_command")
    convert_jax = policy_commands.add_parser(
        "convert-jax",
        help="convert one JAX EMA checkpoint to a checked rollout checkpoint",
    )
    convert_jax.add_argument("--checkpoint", required=True)
    convert_jax.add_argument("--output", required=True)
    convert_jax.add_argument("--norm-stats", required=True)
    convert_jax.add_argument(
        "--parity-config",
        required=True,
        help="matching JAX SFT config used to draw real parity observations",
    )
    convert_jax.add_argument("--parity-batch-size", type=int, default=4)
    convert_jax.add_argument("--parity-execution-horizon", type=int, default=10)
    serve_sft = policy_commands.add_parser(
        "serve-sft",
        help="serve an RLINF SFT checkpoint with RTC support",
    )
    serve_sft.add_argument("--checkpoint", required=True)
    serve_sft.add_argument("--norm-stats", required=True)
    serve_sft.add_argument("--num-steps", type=int, default=10)
    serve_sft.add_argument(
        "--image-preprocessing", choices=("none", "training_fixed_crop"), default="none"
    )
    serve_sft.add_argument("--rtc-guidance-horizon", type=int, default=10)
    serve_sft.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    serve_sft.add_argument(
        "--rtc-prefix-schedule",
        choices=("linear", "exp", "ones", "zeros"),
        default="linear",
    )
    serve_sft.add_argument("--device", default="cuda:0")
    serve_sft.add_argument("--host", default="127.0.0.1")
    serve_sft.add_argument("--port", type=int, default=8080)
    serve_jax = policy_commands.add_parser(
        "serve-jax",
        help="serve the original OpenPI/JAX EMA for chunk reference rollout",
    )
    serve_jax.add_argument("--checkpoint", required=True)
    serve_jax.add_argument("--norm-stats", required=True)
    serve_jax.add_argument("--training-config", required=True)
    serve_jax.add_argument("--num-steps", type=int, default=10)
    serve_jax.add_argument("--image-preprocessing", choices=("none",), default="none")
    serve_jax.add_argument("--host", default="127.0.0.1")
    serve_jax.add_argument("--port", type=int, default=8080)
    serve_groot = policy_commands.add_parser(
        "serve-groot",
        help="serve a GR00T N1.7 checkpoint for synchronous chunk rollout",
    )
    serve_groot.add_argument("--checkpoint", required=True)
    serve_groot.add_argument("--norm-stats", required=True)
    serve_groot.add_argument("--groot-root", required=True)
    serve_groot.add_argument("--checkpoint-sha256", required=True)
    serve_groot.add_argument("--embodiment-tag", default="new_embodiment")
    serve_groot.add_argument("--device", default="cuda:0")
    serve_groot.add_argument("--host", default="127.0.0.1")
    serve_groot.add_argument("--port", type=int, default=5555)
    serve_diffusion = policy_commands.add_parser(
        "serve-diffusion",
        help="serve the trained EMA Diffusion Policy for chunk rollout",
    )
    serve_diffusion.add_argument("--checkpoint", required=True)
    serve_diffusion.add_argument("--norm-stats", required=True)
    serve_diffusion.add_argument("--image-preprocessing", choices=("none",), default="none")
    serve_diffusion.add_argument("--device", default="cuda:0")
    serve_diffusion.add_argument("--host", default="127.0.0.1")
    serve_diffusion.add_argument("--port", type=int, default=8080)
    serve_starvla = policy_commands.add_parser(
        "serve-starvla",
        help="serve an MR0 StarVLA checkpoint over the pickle-TCP protocol",
    )
    serve_starvla.add_argument("--checkpoint", required=True)
    serve_starvla.add_argument("--starvla-root", required=True)
    serve_starvla.add_argument("--device", default="cuda:0")
    serve_starvla.add_argument("--host", default="127.0.0.1")
    serve_starvla.add_argument("--port", type=int, default=8080)
    serve_cfg = policy_commands.add_parser(
        "serve-cfg",
        help="serve an RLINF CFG checkpoint with chunk and RTC support",
    )
    serve_cfg.add_argument("--checkpoint", required=True)
    serve_cfg.add_argument("--norm-stats", required=True)
    serve_cfg.add_argument("--num-steps", type=int, default=5)
    serve_cfg.add_argument(
        "--image-preprocessing", choices=("none", "training_fixed_crop"), default="none"
    )
    serve_cfg.add_argument("--guidance-scale", type=float, default=1.0)
    serve_cfg.add_argument(
        "--guidance-type",
        choices=("positive", "negative", "no_guide"),
        default="positive",
    )
    serve_cfg.add_argument("--positive-only-conditional", action="store_true")
    serve_cfg.add_argument("--rtc-guidance-horizon", type=int, default=10)
    serve_cfg.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    serve_cfg.add_argument(
        "--rtc-prefix-schedule",
        choices=("linear", "exp", "ones", "zeros"),
        default="linear",
    )
    serve_cfg.add_argument(
        "--no-autocast", action="store_true",
        help="use native BF16 parameters without autocast, matching SFT inference",
    )
    serve_cfg.add_argument("--device", default="cuda:0")
    serve_cfg.add_argument("--host", default="127.0.0.1")
    serve_cfg.add_argument("--port", type=int, default=8080)

    benchmark = commands.add_parser(
        "benchmark", help="run, score and summarize the frozen balanced-200 benchmark"
    )
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command")
    benchmark_check = benchmark_commands.add_parser(
        "check-protocol", help="validate and fingerprint one evaluation protocol"
    )
    benchmark_check.add_argument("--protocol", required=True)
    benchmark_template = benchmark_commands.add_parser(
        "make-templates",
        help="free-settle one canonical target for every initial-state asset",
    )
    benchmark_template.add_argument("--states", required=True)
    benchmark_template.add_argument("--protocol", required=True)
    benchmark_template.add_argument("--output", required=True)
    benchmark_template.add_argument("--assets", default="assets")
    benchmark_template.add_argument("--gpus", default="0")
    benchmark_run = benchmark_commands.add_parser(
        "run", help="evaluate one condition over the frozen 800-case panel"
    )
    benchmark_run.add_argument(
        "--config",
        required=True,
        help="managed policy, execution and deployment YAML for one formal condition",
    )
    benchmark_summary = benchmark_commands.add_parser(
        "summarize", help="compare completed conditions over identical cases"
    )
    benchmark_summary.add_argument(
        "--condition",
        action="append",
        required=True,
        help="completed condition directory; repeat in baseline-first order",
    )
    benchmark_score = benchmark_commands.add_parser(
        "score", help="recompute a condition from its saved trajectories"
    )
    benchmark_score.add_argument("--condition", required=True)
    benchmark_score.add_argument("--protocol", required=True)
    benchmark_score.add_argument("--templates", required=True)
    benchmark_score.add_argument(
        "--rescore-from",
        help="original collection protocol YAML; explicitly allow metric/success changes",
    )
    benchmark_score.add_argument(
        "--rescore-templates-from",
        help="original template collection; allow the versioned large-support target correction",
    )
    benchmark_score.add_argument(
        "--output", help="separate score output sharing immutable source trajectories",
    )
    benchmark_score.add_argument(
        "--workers", type=int, default=1, help="offline scoring processes (default: 1)"
    )

    train = commands.add_parser("train", help="run a guarded ClothDojo training workflow")
    train.add_argument(
        "training_kind",
        choices=("openpi-jax", "openpi-rlinf", "recap-value", "steam", "cfg", "diffusion"),
    )
    train.add_argument("--config", required=True, help="path to the root Hydra YAML config")
    train.add_argument(
        "--check",
        action="store_true",
        help="compose and validate the config without launching workers",
    )
    train.add_argument(
        "--gate",
        choices=("initialize", "overfit", "pilot", "parity", "full"),
        help="guarded OpenPI/JAX stage; full requires all four acceptance gates",
    )
    train.add_argument(
        "--advantages-config",
        help="required STEAM labeling config for CFG training",
    )
    train.add_argument(
        "--member-index",
        type=int,
        choices=(0, 1, 2),
        help="independent STEAM ensemble member to train",
    )

    steam = commands.add_parser("steam", help="assemble STEAM evaluator checkpoints")
    steam_commands = steam.add_subparsers(dest="steam_command")
    steam_merge = steam_commands.add_parser(
        "merge", help="merge three independently trained evaluator members"
    )
    steam_merge.add_argument(
        "--member",
        action="append",
        required=True,
        help="member actor checkpoint; repeat exactly three times",
    )
    steam_merge.add_argument("--output", required=True)
    steam_merge.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory",
    )

    data = commands.add_parser("data", help="inspect and export offline datasets")
    data_commands = data.add_subparsers(dest="data_command")
    trim_home = data_commands.add_parser(
        "trim-home-returns",
        help="copy original HIL trajectories and remove final arm-home commands",
    )
    trim_home.add_argument("--source", required=True)
    trim_home.add_argument("--output", required=True)
    trim_home.add_argument("--tolerance", type=float, default=1.0e-6)
    trim_home.add_argument("--min-hold-frames", type=int, default=20)
    trim_home.add_argument("--min-output-actions", type=int, default=50)
    trim_home.add_argument("--workers", type=int, default=8)
    trim_home.add_argument("--plan-only", action="store_true")
    fold_quality = data_commands.add_parser(
        "score-fold-quality", help="score folding completion with the shared Q metric",
    )
    fold_quality.add_argument("--source", required=True, help="trajectory or collection root")
    fold_quality.add_argument("--output", required=True, help="new score output directory")
    fold_quality.add_argument("--assets", default="assets", help="garment assets root")
    fold_quality.add_argument(
        "--reference", required=True,
        help="existing verified flat reference NPZ, or directory containing <asset_id>.npz",
    )
    fold_quality.add_argument(
        "--sample-stride", type=int, default=1, help="state stride; 1 scores every state",
    )
    fold_quality.add_argument("--workers", type=int, default=1, help="CPU scoring workers")
    prepare_recap = data_commands.add_parser(
        "prepare-recap",
        help="validate and freeze expert, HIL rollout, and correction episode semantics",
    )
    prepare_recap.add_argument("--expert", required=True, help="expert collection root")
    prepare_recap.add_argument(
        "--hil-rollout", required=True, help="reviewed canonical HIL rollout root"
    )
    prepare_recap.add_argument(
        "--corrections", required=True, help="canonical HIL correction collection root"
    )
    prepare_recap.add_argument(
        "--output", help="write the immutable JSON plan; omit for validation only"
    )
    prepare_recap_mirrored = data_commands.add_parser(
        "prepare-recap-mirrored",
        help="expand a frozen RECAP plan with canonical original/mirror pairs",
    )
    prepare_recap_mirrored.add_argument("--plan", required=True)
    prepare_recap_mirrored.add_argument(
        "--expert", required=True, help="complete mirrored expert collection"
    )
    prepare_recap_mirrored.add_argument(
        "--hil-rollout", required=True, help="complete mirrored rollout collection"
    )
    prepare_recap_mirrored.add_argument("--output", required=True)
    recap_value_split = data_commands.add_parser(
        "prepare-recap-value-split",
        help="freeze an asset-disjoint value-model train/eval split",
    )
    recap_value_split.add_argument("--plan", required=True)
    recap_value_split.add_argument("--output", required=True)
    recap_value_split.add_argument("--eval-seen-assets", type=int, default=10)
    recap_value_split.add_argument("--eval-unseen-assets", type=int, default=10)
    recap_value_split.add_argument("--seed", type=int, default=260904)
    recap_trajectory_split = data_commands.add_parser(
        "prepare-recap-value-trajectory-split",
        help="freeze a stratified value split with original/mirror pairs kept together",
    )
    recap_trajectory_split.add_argument("--plan", required=True)
    recap_trajectory_split.add_argument("--output", required=True)
    recap_trajectory_split.add_argument("--eval-fraction", type=float, default=0.1)
    recap_trajectory_split.add_argument("--seed", type=int, default=260908)
    recap_returns = data_commands.add_parser(
        "compute-recap-returns",
        help="write RECAP reward/return sidecars for complete value-data exports",
    )
    recap_returns.add_argument("--plan", required=True)
    recap_returns.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        help="expert or HIL rollout export; repeat for every shard",
    )
    recap_returns.add_argument("--tag", default="fail4500_state_v2")
    recap_returns.add_argument("--c-fail-coef", type=float, default=1.0)
    recap_returns.add_argument("--gamma", type=float, default=1.0)
    recap_returns.add_argument(
        "--failure-horizon-steps",
        type=int,
        help=(
            "fixed terminal-failure horizon; must be at least the longest selected "
            "trajectory (defaults to the observed maximum)"
        ),
    )
    recap_returns.add_argument("--overwrite", action="store_true")
    reuse_recap = data_commands.add_parser(
        "reuse-recap-export",
        help="hard-link an original export and bind it to a mirrored RECAP plan",
    )
    reuse_recap.add_argument("--plan", required=True)
    reuse_recap.add_argument("--dataset-root", required=True)
    reuse_recap.add_argument("--output", required=True)
    export_recap = data_commands.add_parser(
        "export-recap",
        help="render one frozen RECAP dataset role as LeRobot v2.1",
    )
    export_recap.add_argument("--plan", required=True, help="ready RECAP plan JSON")
    export_recap.add_argument(
        "--dataset", required=True, choices=("expert", "hil_rollout", "hil_correction")
    )
    export_recap.add_argument("--output", help="new LeRobot dataset directory")
    export_recap.add_argument("--assets", default="assets")
    export_recap.add_argument("--device", default="cuda:0")
    export_recap.add_argument("--cloth-visual", choices=("nocs", "white"), default="nocs")
    export_recap.add_argument("--repo-id")
    export_recap.add_argument("--image-writer-threads", type=int, default=6)
    export_recap.add_argument("--shard-index", type=int)
    export_recap.add_argument("--shard-count", type=int)
    export_recap.add_argument(
        "--pair-member",
        choices=("all", "original", "mirror"),
        default="all",
        help="render all episodes or one member of every canonical mirror pair",
    )
    export_recap.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and inspect the workload without rendering or writing data",
    )
    export = data_commands.add_parser(
        "export-lerobot",
        help="replay selected trajectories as standard LeRobot v2.1",
    )
    merge_shards = data_commands.add_parser(
        "merge-lerobot-shards",
        help="merge disjoint trajectory exports without re-encoding video",
    )
    merge_shards.add_argument("--shard", action="append", required=True)
    merge_shards.add_argument("--output", required=True)
    export.add_argument("--source", required=True, help="source dataset root")
    export.add_argument("--output", help="new LeRobot dataset directory")
    export.add_argument("--assets", default="assets", help="assets used to replay states")
    export.add_argument("--device", default="cuda:0", help="render device for state replay")
    export.add_argument(
        "--cloth-visual",
        choices=("nocs", "white"),
        default="nocs",
        help="garment appearance for paired Newton replay (default: nocs)",
    )
    export.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing transactional staging directory at an episode boundary",
    )
    export.add_argument(
        "--shard-index",
        type=int,
        help="zero-based canonical trajectory shard assigned to this process",
    )
    export.add_argument(
        "--shard-count",
        type=int,
        help="number of disjoint canonical trajectory shards",
    )
    export.add_argument(
        "--isaac-seed",
        type=int,
        help="use Isaac RTX with a deterministic visual layout seed",
    )
    export.add_argument(
        "--visual-assets",
        default="data/visual-assets",
        help="Poly Haven materials and optional rooms prepared for Isaac rendering",
    )
    export.add_argument(
        "--include-room",
        action="store_true",
        help="include each sampled RoboDojo room; requires --isaac-seed",
    )
    from clothdojo.assets.visual_benchmark import VISUAL_PROFILES
    from clothdojo.data.isaac_render import parse_visual_randomization

    export.add_argument(
        "--visual-profile",
        choices=VISUAL_PROFILES,
        help="fixed training or held-out appearance pools; requires --isaac-seed",
    )

    export.add_argument(
        "--randomize",
        type=parse_visual_randomization,
        metavar="COMPONENTS",
        help=(
            "Isaac components varied per episode: none, all, or a comma-separated "
            "subset of cloth,table,ground,lighting,background (default: all)"
        ),
    )
    export.add_argument("--repo-id", help="LeRobot repository identity stored in metadata")
    export.add_argument(
        "--image-writer-threads",
        type=int,
        default=6,
        help="threads used by the pinned LeRobot image writer",
    )
    export.add_argument(
        "--min-segment-frames",
        type=int,
        default=DEFAULT_MIN_SEGMENT_FRAMES,
        help="drop reviewed-good fragments shorter than this many frames",
    )
    export.add_argument(
        "--whole-trajectory-good-only",
        action="store_true",
        help="keep only trajectories explicitly reviewed Good as a whole",
    )
    export.add_argument(
        "--plan-only",
        action="store_true",
        help="inspect the export plan without decoding, rendering, or writing data",
    )
    prepare_isaac = data_commands.add_parser(
        "prepare-isaac-assets",
        help="prepare curated Poly Haven visuals used by Isaac export",
    )
    prepare_isaac.add_argument(
        "--output",
        default="data/visual-assets",
        help="visual asset directory",
    )
    prepare_isaac.add_argument(
        "--include-room",
        action="store_true",
        help="also download the optional room assets",
    )
    mirror = data_commands.add_parser(
        "augment-mirror",
        help="copy a canonical Piper collection and add bilateral mirror trajectories",
    )
    mirror.add_argument("--source", required=True, help="canonical Piper collection root")
    mirror.add_argument("--output", required=True, help="new independent collection root")
    canonicalize = data_commands.add_parser(
        "canonicalize-actions",
        help="copy a collection and replace actions with runtime targets",
    )
    canonicalize.add_argument("--source", required=True, help="canonical Piper collection root")
    canonicalize.add_argument("--output", help="new action-overlay collection root")
    canonicalize.add_argument(
        "--plan-only",
        action="store_true",
        help="inspect contracts and changes without writing an overlay",
    )
    retarget = data_commands.add_parser(
        "retarget",
        help="retarget Piper trajectories to another robot",
    )
    retarget.add_argument("--source", required=True, help="trajectory or collection root")
    retarget.add_argument("--output", required=True, help="new retarget artifact root")
    retarget.add_argument(
        "--target-robot",
        required=True,
        choices=(
            "aloha2",
            "x5",
            "fr3_duo",
            "ur5e_2f85",
        ),
        help="fixed target robot profile",
    )
    retarget.add_argument(
        "--assets",
        default="assets",
        help="project asset directory",
    )
    retarget_devices = retarget.add_mutually_exclusive_group()
    retarget_devices.add_argument(
        "--device",
        default="cuda:0",
        help="Newton IK and rigid-dynamics device",
    )
    retarget_devices.add_argument(
        "--devices",
        nargs="+",
        help="retarget GPUs, e.g. cuda:0 cuda:1",
    )
    retarget.add_argument(
        "--workers-per-device",
        type=int,
        default=1,
        help="persistent processes per GPU (default: 1); use NVIDIA MPS for same-GPU concurrency",
    )
    review = data_commands.add_parser(
        "render-review",
        help="incrementally render lightweight trajectory review videos",
    )
    review.add_argument("--source", required=True, help="collection or rollout root")
    review.add_argument("--assets", default="assets", help="assets used for replay")
    review.add_argument("--device", default="cuda:0", help="Newton render device")
    review.add_argument(
        "--camera",
        choices=("cam_high", "cam_low"),
        default="cam_high",
        help="native workstation camera to render",
    )
    review.add_argument(
        "--output",
        help="independent review output root; defaults to SOURCE/review",
    )
    review.add_argument(
        "--trajectory",
        action="append",
        help="render only this trajectory ID; repeat to select multiple trajectories",
    )
    review.add_argument(
        "--fps",
        type=int,
        default=10,
        help="review frame rate; must divide the rollout control rate",
    )
    norm_stats = data_commands.add_parser(
        "compute-norm-stats",
        help="compute OpenPI stats using the ALOHA transforms used by training",
    )
    norm_stats.add_argument("--dataset", required=True, help="good-only LeRobot root")
    norm_stats.add_argument("--output", required=True, help="new external stats directory")
    norm_stats.add_argument("--block-size", type=int, default=4096)
    norm_stats.add_argument(
        "--action-chunk-boundary",
        choices=("repeat_last", "full_only"),
        default="repeat_last",
        help="episode-boundary supervision policy; must match the training config",
    )
    collect = commands.add_parser(
        "collect",
        help="run teleoperation, review, HIL and asset-curation workflows",
    )
    collect_commands = collect.add_subparsers(dest="collect_command")
    collect_serve = collect_commands.add_parser(
        "serve",
        help="serve the local collection console",
    )
    collect_serve.add_argument(
        "--config",
        default="configs/collection.yaml",
        help="deployment-specific collection configuration",
    )
    return parser


def _comma_separated(value: str, name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise ValueError(f"{name} must be a non-empty comma-separated list without duplicates")
    return items
