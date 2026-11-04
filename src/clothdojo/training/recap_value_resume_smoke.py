"""Audit a native RLinf RECAP resume before committing another training run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from clothdojo.training.recap_value import (
    inspect_recap_value_training,
    write_recap_value_launch_manifest,
)
from clothdojo.training.rlinf import _load_runtime, compose_config


def _source_validation(resume_dir: Path) -> dict[str, float]:
    events = EventAccumulator(str(resume_dir.parents[2] / "tensorboard")).Reload()
    completed_steps = int(resume_dir.name.removeprefix("global_step_"))
    event_step = completed_steps - 1
    result: dict[str, float] = {}
    for name in ("loss", "mae", "value_spearman"):
        points = {event.step: float(event.value) for event in events.Scalars(f"eval/{name}")}
        if event_step not in points:
            raise RuntimeError(f"source validation is missing eval/{name} at {event_step}")
        result[name] = points[event_step]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    cfg = compose_config(args.config, [f"runner.logger.log_path={output}"])
    manifest = inspect_recap_value_training(cfg)
    resume = manifest.get("resume")
    if not isinstance(resume, dict):
        raise RuntimeError("resume smoke requires a checked native resume checkpoint")
    manifest["purpose"] = "native_rlinf_resume_load_eval_and_one_update"
    write_recap_value_launch_manifest(cfg, manifest)

    runtime = _load_runtime("recap-value")
    cfg = runtime.validate_cfg(cfg)
    runtime.multiprocessing.set_start_method("spawn", force=True)
    cluster = runtime.cluster_type(cluster_cfg=cfg.cluster)
    from clothdojo.training.worker_environment import configure_cluster_worker_environment

    configure_cluster_worker_environment(cluster, cfg)
    placement = runtime.placement_type(cfg, cluster)
    actor = runtime.worker_type.create_group(cfg).launch(
        cluster,
        name="ReCapValueResumeSmoke",
        placement_strategy=placement.get_strategy("actor"),
    )
    actor.init_worker().wait()

    resume_dir = Path(str(resume["path"]))
    actor.load_checkpoint(str(resume_dir / "actor")).wait()
    completed_steps = int(resume["completed_steps"])
    actor.set_global_step(completed_steps).wait()
    validation = actor.run_eval().wait()[0]
    expected = _source_validation(resume_dir)
    differences = {
        name: abs(float(validation[name]) - value) for name, value in expected.items()
    }
    if any(delta > 2e-5 for delta in differences.values()):
        raise RuntimeError(
            f"restored validation differs from the source checkpoint: {differences}"
        )

    training = actor.run_training().wait()[0]
    required = ("loss", "grad_norm", "lr")
    if any(not math.isfinite(float(training[name])) for name in required):
        raise RuntimeError(f"non-finite resumed update: {training}")
    if not math.isclose(float(training["lr"]), 1e-6, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"resumed backbone LR changed: {training['lr']}")

    report = {
        "status": "passed",
        "resume": resume,
        "source_validation": expected,
        "restored_validation": {name: float(validation[name]) for name in expected},
        "validation_absolute_difference": differences,
        "first_update": {name: float(training[name]) for name in required},
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
