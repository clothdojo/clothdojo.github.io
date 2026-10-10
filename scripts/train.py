"""Fine-tune Pi0.5 on a Generated RGB LeRobot dataset."""

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="validate inputs without starting Ray or training")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("reference training requires Linux x86_64")
    if args.output.exists() or args.steps < 1 or args.gpu < 0:
        parser.error("use a fresh output, positive steps and a non-negative GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # This reference uses one local GPU. Avoid inherited cluster NCCL plugins
    # and fabric settings, which can hang even a one-rank parameter broadcast.
    os.environ.update(NCCL_NET_PLUGIN="none", NCCL_TUNER_PLUGIN="none",
                      NCCL_PROFILER_PLUGIN="none", NCCL_IB_DISABLE="1",
                      NCCL_P2P_DISABLE="1", NCCL_SOCKET_IFNAME="lo")
    model, dataset, stats, output = (path.expanduser().resolve() for path in
                                     (args.model, args.dataset, args.norm_stats, args.output))
    policy = json.loads((model / "policy.json").read_text())
    provenance = json.loads((stats / "clothdojo.json").read_text())
    info = json.loads((dataset / "meta/info.json").read_text())
    sidecar = json.loads((dataset / "meta/clothdojo.json").read_text())
    if policy.get("schema") != "clothdojo_pi05_policy_v1" or policy.get("task") not in {"shirt_flatten", "shirt_fold"}:
        parser.error("expected a supported Pi0.5 model bundle")
    source_visual = policy.get("cloth_visual")
    if source_visual not in {"nocs", "white", "rgb"}:
        parser.error("expected a NOCS, white, or RGB model bundle")
    target_visual = "rgb"
    if info.get("codebase_version") != "v2.1" or info.get("fps") != 50:
        parser.error("expected a 50 Hz LeRobot v2.1 dataset")
    for key in ("observation.state", "action"):
        if info.get("features", {}).get(key, {}).get("shape") != [14]:
            parser.error(f"expected a 14-dimensional {key}")
    for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        if f"observation.images.{name}" not in info["features"]:
            parser.error(f"dataset is missing {name}")
    for field, filename in (("dataset_info_sha256", "info.json"),
                            ("dataset_sidecar_sha256", "clothdojo.json")):
        if provenance.get(field) != digest(dataset / "meta" / filename):
            parser.error(f"normalization does not match this dataset: {field}")
    if (provenance.get("config_name") != "pi05_aloha_robotwin"
            or provenance.get("action_horizon") != 50
            or provenance.get("data_kwargs", {}).get("adapt_to_pi") is not False
            or provenance.get("data_kwargs", {}).get("extra_delta_transform") is not True):
        parser.error("normalization has an incompatible action contract")
    scenario = sidecar.get("scenario", {})
    dataset_task = scenario.get("task_name") or scenario.get("scenario_id")
    if dataset_task != policy["task"]:
        parser.error("dataset and model tasks differ")
    appearance = (sidecar.get("observation") or {}).get("cloth_visual")
    if appearance is None:
        appearance = (sidecar.get("rendering") or {}).get("cloth_visual")
    if appearance is None:
        parser.error("RGB dataset must declare observation.cloth_visual=rgb")
    if appearance is not None and appearance != target_visual:
        parser.error("dataset and model appearances differ")
    tokenizer = model / "cache/big_vision/paligemma_tokenizer.model"
    if digest(tokenizer) != policy["tokenizer_sha256"]:
        parser.error("tokenizer checksum differs")
    if "initial_states_sha256" in policy and digest(model / "initial_states.npz") != policy["initial_states_sha256"]:
        parser.error("initial-state checksum differs")
    from omegaconf import OmegaConf

    from clothdojo.rollout.checkpoint import inspect_rlinf_openpi_checkpoint
    from clothdojo.training.openpi import validate_action_chunk_boundary

    inspection = inspect_rlinf_openpi_checkpoint(model / "weights", action_horizon=50)
    if inspection["weights_sha256"] != policy["weights_sha256"]:
        parser.error("initial model checksum differs")
    boundary = validate_action_chunk_boundary(provenance["action_chunk_boundary"])
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs/training/pi05.yaml")
    cfg.clothdojo.update(task=policy["task"], cloth_visual=target_visual,
                         action_chunk_boundary=boundary,
                         initial_states=str(model / "initial_states.npz") if (model / "initial_states.npz").is_file() else None)
    cfg.data.train_data_paths = str(dataset)
    cfg.data.openpi_data_home = str(model / "cache")
    cfg.actor.model.model_path = str(model / "weights")
    cfg.actor.model.openpi_data.norm_stats_path = str(stats / "norm_stats.json")
    cfg.actor.model.openpi_data.default_prompt = provenance["data_kwargs"]["default_prompt"]
    cfg.runner.logger.log_path = str(output)
    cfg.runner.max_steps = cfg.runner.save_interval = args.steps
    cfg.actor.optim.total_training_steps = args.steps
    cfg.actor.optim.lr_warmup_steps = min(100, max(1, args.steps // 10))
    if args.check:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    os.environ["OPENPI_DATA_HOME"] = str(model / "cache")
    import ray

    try:
        from rlinf.config import validate_cfg
        from rlinf.runners.sft_runner import SFTRunner
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement

        from clothdojo.training.sft import Pi05SftWorker

        from unittest.mock import patch

        # RLinf must initialize Ray itself to set up its managers. Override
        # only that initialization's connection target, so its default "auto"
        # cannot attach this single-user run to somebody else's cluster.
        initialize_ray = ray.init

        def initialize_local_ray(**kwargs):
            kwargs.update(address="local", num_cpus=2, num_gpus=1,
                          object_store_memory=128 * 1024 * 1024,
                          include_dashboard=True, dashboard_host="127.0.0.1", dashboard_port=0,
                          _temp_dir=tempfile.mkdtemp(prefix="clothdojo-sft-"))
            return initialize_ray(**kwargs)

        with patch.object(ray, "init", initialize_local_ray):
            cfg = validate_cfg(cfg)
        cluster = Cluster(cluster_cfg=cfg.cluster)
        placement = HybridComponentPlacement(cfg, cluster)
        actors = Pi05SftWorker.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=placement.get_strategy("actor"))
        runner = SFTRunner(cfg=cfg, actor=actors)
        runner.init_workers()
        runner.run()
    finally:
        ray.shutdown()
    bundle = output / "pi05_sft/checkpoints" / f"global_step_{args.steps}" / "actor"
    if not (bundle / "policy.json").is_file():
        raise RuntimeError("training did not publish its final model bundle")
    print(bundle)


if __name__ == "__main__":
    main()
