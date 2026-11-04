"""Run two real updates and one native RLinf RECAP validation pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from clothdojo.training.recap_value import (
    inspect_recap_value_training,
    write_recap_value_launch_manifest,
)
from clothdojo.training.rlinf import _load_runtime, compose_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    cfg = compose_config(args.config, [f"runner.logger.log_path={output}"])
    manifest = inspect_recap_value_training(cfg)
    manifest["purpose"] = "native_rlinf_two_updates_and_validation"
    manifest["smoke_eval_max_samples_per_shard"] = 16
    write_recap_value_launch_manifest(cfg, manifest)

    # The full held-out set remains unchanged for Stage 1/2. The smoke only
    # needs one distributed microbatch from every non-empty validation shard
    # to exercise decoding, collation, forward metrics, and collectives.
    for entry in cfg.data.eval_data_paths:
        OmegaConf.update(entry, "max_samples", 16, force_add=True)

    runtime = _load_runtime("recap-value")
    cfg = runtime.validate_cfg(cfg)
    runtime.multiprocessing.set_start_method("spawn", force=True)
    cluster = runtime.cluster_type(cluster_cfg=cfg.cluster)
    from clothdojo.training.worker_environment import configure_cluster_worker_environment

    configure_cluster_worker_environment(cluster, cfg)
    placement = runtime.placement_type(cfg, cluster)
    actor = runtime.worker_type.create_group(cfg).launch(
        cluster,
        name="ReCapValueSmoke",
        placement_strategy=placement.get_strategy("actor"),
    )
    actor.init_worker().wait()

    train_metrics = []
    for step in range(2):
        actor.set_global_step(step).wait()
        metrics = actor.run_training().wait()[0]
        train_metrics.append(metrics)
        print(json.dumps({"step": step + 1, "train": metrics}), flush=True)

    validation = actor.run_eval().wait()[0]
    checkpoint = output / "checkpoint" / "actor"
    actor.save_checkpoint(str(checkpoint), 2).wait()
    report = {
        "status": "passed",
        "worker": "rlinf.workers.sft.fsdp_value_sft_worker.FSDPValueSftWorker",
        "train": train_metrics,
        "validation": validation,
        "checkpoint": str(checkpoint),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
