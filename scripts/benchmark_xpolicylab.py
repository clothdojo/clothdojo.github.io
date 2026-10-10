"""Evaluate one XPolicyLab model across a fixed ClothDojo initial-state pool."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="XPolicyLab policy directory name")
    parser.add_argument("--model-id", required=True, help="stable checkpoint or run identifier")
    parser.add_argument("--initial-states", type=Path,
                        help="default: scripts/data/initial_states/<task>.npz")
    parser.add_argument("--template", type=Path,
                        help="default: templates/rand_ZZCWITH5HY.npz for the included garment")
    parser.add_argument("--task", choices=("shirt_flatten", "shirt_fold"), default="shirt_flatten")
    parser.add_argument("--cloth-visual", choices=("nocs", "white", "rgb"), default="nocs")
    parser.add_argument("--isaac-gpu", help="Isaac GPU ordinal for live RGB observations")
    parser.add_argument("--visual-assets", type=Path, default=Path("assets/visual"),
                        help="downloaded Isaac material assets (default: assets/visual)")
    parser.add_argument("--visual-profile", choices=("train", "test_cloth", "test_background", "test_lighting", "test_all"), default="train")
    parser.add_argument("--visual-seed", type=int, default=42)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path,
                        help="default: outputs/evaluations/<task>/<model-id>")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, help="number of consecutive initial states; default: all remaining")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, help="default: the task's scoring horizon")
    parser.add_argument("--protocol", type=Path, help="override the task's scoring protocol")
    parser.add_argument("--server", default="ws://127.0.0.1:6000")
    parser.add_argument("--xpl-root", type=Path, default=Path("XPolicyLab"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.initial_states is None:
        args.initial_states = Path("scripts/data/initial_states") / f"{args.task}.npz"
    if args.template is None:
        args.template = Path("templates/rand_ZZCWITH5HY.npz")
    if args.output is None:
        if args.model_id in {".", ".."} or not args.model_id.isascii() or not all(
            char.isalnum() or char in "-_." for char in args.model_id
        ):
            parser.error("model id must use ASCII letters, digits, '-', '_' or '.' for the default output; otherwise pass --output")
        args.output = Path("outputs/evaluations") / args.task / args.model_id

    if not args.model_id.strip() or args.seed < 0 or args.start_index < 0:
        parser.error("model id must be non-empty; seed and start index must be non-negative")
    if args.count is not None and args.count < 1:
        parser.error("count must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.cloth_visual == "rgb" and args.isaac_gpu is None:
        parser.error("RGB evaluation requires --isaac-gpu")
    if not args.initial_states.is_file() or not args.template.is_file():
        parser.error("initial-state pool and canonical template must exist")

    from clothdojo.evaluation.metric import load_canonical_template
    from clothdojo.evaluation.protocol import load_evaluation_protocol
    from clothdojo.rollout.initial_states import InitialStatePool

    default_protocol = Path(__file__).resolve().parents[1] / "configs/evaluation" / f"{args.task}.yaml"
    protocol_path = args.protocol or default_protocol
    protocol = load_evaluation_protocol(protocol_path)
    pool = InitialStatePool.load(args.initial_states)
    template = load_canonical_template(args.template)
    if protocol.task_name != args.task:
        parser.error("scoring protocol task differs from the requested task")
    if (template.asset_id != pool.metadata.get("garment_id") or
            template.asset_sha256 != pool.metadata.get("garment_sha256") or
            template.particle_count != pool.particle_count):
        parser.error("template and initial-state pool identities differ")
    count = args.count if args.count is not None else pool.state_count - args.start_index
    if count < 1 or args.start_index + count > pool.state_count:
        parser.error("requested initial-state range exceeds the pool")
    steps = args.steps or protocol.horizon_steps
    if steps > protocol.horizon_steps:
        parser.error("steps cannot exceed the task's scoring horizon")
    report = {
        "schema": "clothdojo_xpolicylab_benchmark_v1",
        "status": "running",
        "task": args.task,
        "policy_name": args.policy,
        "model_id": args.model_id,
        "initial_states_sha256": hashlib.sha256(args.initial_states.read_bytes()).hexdigest(),
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "requested_episodes": count,
        "steps_per_episode": steps,
        "episodes": [],
        "successes": None,
        "success_rate": None,
    }
    args.output.mkdir(parents=True)

    def publish() -> None:
        temporary = args.output / ".benchmark.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "benchmark.json")

    publish()
    example = Path(__file__).resolve().parent / "eval_xpolicylab.py"
    try:
        for index in range(args.start_index, args.start_index + count):
            episode = f"episode_{index:06d}"
            trajectory = args.output / "episodes" / episode
            seed = args.seed + index
            command = [
                sys.executable, str(example), "--policy", args.policy,
                "--model-id", args.model_id, "--initial-states", str(args.initial_states),
                "--initial-index", str(index), "--template", str(args.template),
                "--protocol", str(protocol_path), "--task", args.task,
                "--cloth-visual", args.cloth_visual, "--assets", str(args.assets),
                "--output", str(trajectory), "--steps", str(steps),
                "--seed", str(seed), "--server", args.server,
                "--xpl-root", str(args.xpl_root), "--device", args.device,
            ]
            if args.cloth_visual == "rgb":
                command += ["--isaac-gpu", args.isaac_gpu,
                            "--visual-assets", str(args.visual_assets),
                            "--visual-profile", args.visual_profile,
                            "--visual-seed", str(args.visual_seed)]
            log_path = args.output / f"{episode}.log"
            with log_path.open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise RuntimeError(
                    f"{episode} exited with status {result.returncode}; see {log_path}"
                )
            summary = json.loads((trajectory / "summary.json").read_text())
            score = summary["score"]
            if (summary["model_id"] != args.model_id or
                    summary["states"] != steps + 1 or summary["actions"] != steps or
                    score["protocol_fingerprint"] != protocol.fingerprint):
                raise RuntimeError(f"inconsistent episode result: {episode}")
            report["episodes"].append({
                "initial_index": index,
                "seed": seed,
                "trajectory": f"episodes/{episode}",
                "score": score,
            })
            publish()
        successes = [row["score"]["success"] for row in report["episodes"]]
        if all(value is not None for value in successes):
            report["successes"] = sum(value is True for value in successes)
            report["success_rate"] = report["successes"] / count
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        publish()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
