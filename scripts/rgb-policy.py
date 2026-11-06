"""Public RGB Flatten: download, train, evaluate through the existing ClothDojo CLI."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/clothdojo").is_dir())
DATA_REV = "a498929e3f4a1f070cb820d48752cd8eb025752e"
ASSET_REV = "22923d5d4e5c628b7539e478bb27f0fd79e820f3"


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + "\n")


def install_visuals(work):
    """Expose downloaded assets at the existing renderer's standard location."""
    source = work / "assets/data/visual-assets"
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = ROOT / "data/visual-assets" / path.relative_to(source)
        if target.exists():
            with path.open("rb") as left, target.open("rb") as right:
                if (
                    hashlib.file_digest(left, "sha256").digest()
                    != hashlib.file_digest(right, "sha256").digest()
                ):
                    raise ValueError(f"existing visual asset differs: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)


def check_visual_dependencies(work):
    missing, different = [], []
    for row in read(work / "assets/upstream_visual_dependencies.json"):
        path = ROOT / row["path"]
        if not path.is_file():
            missing.append(row["path"])
            continue
        if path.stat().st_size != row["bytes"]:
            different.append(row["path"])
            continue
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                different.append(row["path"])
    if missing or different:
        raise RuntimeError(
            f"Restore upstream visuals before evaluation: {len(missing)} missing, "
            f"{len(different)} different. See docs/upstream-dependencies.md. "
            f"First affected path: {(missing + different)[0]}"
        )


def prepare(work, smoke):
    if read(work / "download.json")["smoke_only"] != smoke:
        raise ValueError("use separate directories for smoke and full runs")
    cfg = yaml.safe_load((ROOT / "configs/training/rgb_flatten_diffusion.yaml").read_text())
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg.update(
        datasets=[str(work / "dataset")],
        micro_batch_size=1 if smoke else 8,
        global_batch_size=1 if smoke else 8,
        num_workers=0 if smoke else 4,
        max_steps=2 if smoke else 80000,
        log_interval=1 if smoke else 20,
        save_interval=2 if smoke else 5000,
    )
    (work / "train.yaml").write_text(yaml.safe_dump(cfg))
    condition = work / "assets/benchmark/conditions/rgb-flatten"
    states = condition / "states"
    meta = read(states / "meta.json")
    panel = meta["selection_provenance"]["panel_id"]
    if smoke:
        from dataclasses import replace

        from clothdojo.rollout.initial_states import InitialStatePool

        row = next(item.copy() for item in meta["assets"] if item["split"] == "seen")
        pool = InitialStatePool.load(states / row["path"])
        states = work / "smoke-state"
        path = states / row["path"]
        if not path.exists():
            replace(
                pool,
                particle_q=pool.particle_q[:1],
                particle_qd=pool.particle_qd[:1],
                seeds=pool.seeds[:1],
                metadata={
                    key: value[:1]
                    if isinstance(value, list) and len(value) == pool.state_count
                    else value
                    for key, value in pool.metadata.items()
                },
            ).save(path)
        row.update(state_count=1, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        panel = "public_rgb_flatten_smoke_only"
        meta.update(
            assets=[row],
            seen_assets=1,
            unseen_assets=0,
            states_per_asset=1,
            physical_states_per_asset=1,
            selection_provenance={"panel_id": panel},
        )
        write(states / "meta.json", meta)
    scenario = read(condition / "scenario.json")
    scenario.update(scenario_id="public_rgb_flatten_diffusion", action_horizon=8)
    write(work / "scenario.json", scenario)
    write(
        work / "benchmark.json",
        {
            "schema": "clothdojo_frozen_benchmark",
            "panel_id": panel,
            "scenario": str(work / "scenario.json"),
            "protocol": str(condition / "protocol.json"),
            "states": str(states),
            "templates": str(condition / "templates"),
            "assets": str(work / "assets/benchmark/assets"),
        },
    )
    write(
        work / "deployment.json",
        {
            "schema": "clothdojo_evaluation_deployment",
            "policy": {"nodes": [{"host": "local", "gpus": [0], "port_start": 28000}]},
            "simulation": {"nodes": [{"host": "local", "gpus": [1]}]},
            "score_workers": 1,
            "startup_timeout_seconds": 1800,
        },
    )
    checkpoint = work / "train" / ("step_000002.pt" if smoke else "step_080000.pt")
    write(
        work / "evaluate.json",
        {
            "schema": "clothdojo_benchmark_run",
            "condition_id": panel + "_diffusion",
            "output": str(work / ("evaluation-smoke" if smoke else "evaluation")),
            "benchmark": str(work / "benchmark.json"),
            "deployment": str(work / "deployment.json"),
            "policy": {
                "kind": "xpolicy",
                "checkpoint": str(checkpoint),
                "model": "diffusion",
            },
            "execution": {
                "mode": "chunk",
                "exec_horizon": 8,
                "batch_size": 1 if smoke else 4,
                "cloth_visual": "rgb",
                "visual_profiles": ["train", "test_all"],
                "visual_seed": 20260911,
                "image_preprocessing": "none",
            },
        },
    )


def configure_model(work, model, checkpoint=None, output=None):
    """Bind each checkpoint to its own resumable output; never mix model results."""
    from clothdojo.rollout.xpolicy import artifact_sha256, validate_model_name
    from clothdojo.runtime_identity import runtime_source_sha256

    validate_model_name(model)

    path = work / "evaluate.json"
    run = read(path)
    checkpoint = (checkpoint or Path(run["policy"]["checkpoint"])).expanduser().resolve()
    policy = {"kind": "xpolicy", "model": model, "checkpoint": str(checkpoint)}
    if checkpoint.exists():
        digest = artifact_sha256(checkpoint)
        code = runtime_source_sha256(ROOT / "src/clothdojo")[:8]
        policy["expected_checkpoint_sha256"] = digest
        name = model.replace(":", "-").replace(".", "-")[:40]
        run["condition_id"] = f"rgb_flatten_{name}_{digest[:12]}_{code}"
        run["output"] = str(work / "evaluations" / f"{name}-{digest[:12]}-{code}")
        if model == "diffusion":
            import torch

            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("schema") != "clothdojo_diffusion_v1":
                raise ValueError("diffusion requires a ClothDojo-trained Diffusion checkpoint")
            horizon = int(saved["config"]["n_action_steps"])
            run["execution"]["exec_horizon"] = horizon
            scenario_path = Path(read(work / "benchmark.json")["scenario"])
            scenario = read(scenario_path)
            scenario["action_horizon"] = horizon
            write(scenario_path, scenario)
    run["policy"] = policy
    if output is not None:
        run["output"] = str(output.expanduser().resolve())
    write(path, run)
    return run


def check_xpolicy_runtime():
    try:
        importlib.import_module("client_server.ws.model_client")
        importlib.import_module("client_server.ws.model_server")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Install the optional policy runtime first: bash scripts/setup.sh --xpolicy"
        ) from error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["download", "train", "evaluate"])
    p.add_argument("--work", type=Path)
    p.add_argument(
        "--model", default="diffusion", help="registered model or importable module:Model"
    )
    p.add_argument("--checkpoint", type=Path, help="self-contained checkpoint to evaluate")
    p.add_argument("--output", type=Path, help="new evaluation output directory (optional)")
    p.add_argument(
        "--assets-only",
        action="store_true",
        help="download evaluation assets without training data",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="one episode, two updates, one evaluation state; not benchmark results",
    )
    p.add_argument("--check", action="store_true", help="validate configuration without GPU work")
    a = p.parse_args()
    if a.assets_only and a.stage != "download":
        p.error("--assets-only is a download option")
    if a.stage != "evaluate" and (
        a.model != "diffusion" or a.checkpoint is not None or a.output is not None
    ):
        p.error("--model, --checkpoint and --output are evaluation options")
    if a.check and a.stage == "download":
        p.error("--check is a train/evaluate option")
    work = (
        a.work or ROOT / "outputs" / ("rgb-policy-smoke" if a.smoke else "rgb-policy")
    ).resolve()
    if not work.is_relative_to(ROOT):
        p.error("--work must be inside the project for managed evaluation")
    if a.stage == "download":
        # The pinned training client otherwise requests a fresh Xet token for every
        # small asset, exhausting Hub API quota before the asset tree is fetched.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        from huggingface_hub import snapshot_download

        patterns = (
            [
                "meta/*",
                "data/chunk-000/episode_000000.parquet",
                "videos/chunk-000/*/episode_000000.mp4",
                "README.md",
            ]
            if a.smoke
            else None
        )
        if not a.assets_only:
            snapshot_download(
                "clothdojo/flatten",
                repo_type="dataset",
                revision=DATA_REV,
                local_dir=work / "dataset",
                allow_patterns=patterns,
                max_workers=8,
            )
        snapshot_download(
            "clothdojo/assets",
            repo_type="dataset",
            revision=ASSET_REV,
            local_dir=work / "assets",
            max_workers=8,
            allow_patterns=[
                "assets/**",
                "benchmark/**",
                "data/**",
                "templates/**",
                "licenses/**",
                "*LICENSE*",
                "*NOTICES*",
                "upstream_visual_dependencies.json",
            ],
        )
        if not (work / "assets/benchmark/conditions/rgb-flatten/states/meta.json").is_file():
            raise RuntimeError(
                "asset download is incomplete; retry after the Hub rate limit resets"
            )
        if a.smoke and not a.assets_only:
            for name in ("episodes.jsonl", "episodes_stats.jsonl"):
                path = work / "dataset/meta" / name
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                path.write_text(
                    "".join(json.dumps(r) + "\n" for r in rows if r["episode_index"] == 0)
                )
            info = read(work / "dataset/meta/info.json")
            ep = read(work / "dataset/meta/episodes.jsonl")
            info.update(
                total_episodes=1,
                total_frames=ep["length"],
                total_chunks=1,
                total_videos=4,
                splits={"train": "0:1"},
            )
            write(work / "dataset/meta/info.json", info)
        write(
            work / "download.json",
            {
                "smoke_only": a.smoke,
                "dataset_revision": DATA_REV,
                "assets_revision": ASSET_REV,
                "training_data_included": not a.assets_only,
            },
        )
        install_visuals(work)
    if a.stage == "download" or not (work / "evaluate.json").exists():
        prepare(work, a.smoke)
    elif read(work / "download.json")["smoke_only"] != a.smoke:
        raise ValueError("use separate directories for smoke and full runs")
    cli = [sys.executable, "-m", "clothdojo"]
    if a.stage == "train":
        if not (work / "dataset/meta/info.json").is_file():
            raise ValueError("download the training data without --assets-only before training")
        command = [*cli, "train", "diffusion", "--config", str(work / "train.yaml")]
        command += (
            ["--check", "--project", str(ROOT)]
            if a.check
            else ["--project", str(ROOT), "--output", str(work / "train")]
        )
        subprocess.run(command, cwd=ROOT, check=True)
    elif a.stage == "evaluate":
        check_xpolicy_runtime()
        check_visual_dependencies(work)
        run = configure_model(work, a.model, a.checkpoint, a.output)
        if not a.check and not Path(run["policy"]["checkpoint"]).exists():
            raise FileNotFoundError(f"checkpoint is missing: {run['policy']['checkpoint']}")
        if a.check:
            from clothdojo.evaluation.run_config import load_benchmark_run_config

            cfg = load_benchmark_run_config(work / "evaluate.json", project_root=ROOT)
            for path in (
                cfg.benchmark.states / "meta.json",
                cfg.benchmark.protocol,
                cfg.benchmark.scenario,
                cfg.benchmark.templates / "meta.json",
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)
            print(
                json.dumps(
                    {
                        "status": "configuration_checked",
                        "checkpoint_exists": cfg.policy.checkpoint.is_file(),
                        "smoke_only": a.smoke,
                    }
                )
            )
        else:
            subprocess.run(
                [*cli, "benchmark", "run", "--config", str(work / "evaluate.json")],
                cwd=ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
