"""Generate a bounded collection and select successes with the frozen task protocol."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--template", type=Path,
                        help="default: templates/<garment-id>.npz from the initial-state pool")
    parser.add_argument("--initial-states", type=Path,
                        help="use a generated pool instead of the model bundle's pool")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path,
                        help="default: outputs/generated/<task>")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--successes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("chunk", "rtc"), default="chunk")
    parser.add_argument("--full-state", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.successes <= args.attempts or args.seed < 0:
        parser.error("use a non-negative seed and 1 <= successes <= attempts")
    from clothdojo.evaluation.metric import load_canonical_template
    from clothdojo.evaluation.protocol import load_evaluation_protocol
    from clothdojo.rollout.initial_states import InitialStatePool

    task = json.loads((args.model / "policy.json").read_text())["task"]
    if task not in {"shirt_flatten", "shirt_fold"}:
        parser.error("unsupported task")
    if args.output is None:
        args.output = Path("outputs/generated") / task
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; use --output for another run")
    scripts = Path(__file__).resolve().parent
    protocol_path = scripts.parent / "configs/evaluation" / f"{task}.yaml"
    protocol = load_evaluation_protocol(protocol_path)
    initial_states = args.initial_states or args.model / "initial_states.npz"
    pool = InitialStatePool.load(initial_states)
    if pool.state_count < 1:
        parser.error("generation requires a non-empty initial-state pool")
    if args.template is None:
        garment_id = pool.metadata.get("garment_id")
        if not isinstance(garment_id, str) or not garment_id:
            parser.error("initial-state pool has no garment id; pass --template")
        args.template = Path("templates") / f"{garment_id}.npz"
    template = load_canonical_template(args.template)
    if (template.asset_id != pool.metadata.get("garment_id")
            or template.asset_sha256 != pool.metadata.get("garment_sha256")
            or template.particle_count != pool.particle_count):
        parser.error("template and initial-state pool identities differ")
    args.output.mkdir(parents=True)
    report = {"status": "running", "task": task, "protocol_fingerprint": protocol.fingerprint,
              "initial_states_sha256": hashlib.sha256(initial_states.read_bytes()).hexdigest(),
              "attempt_limit": args.attempts, "requested_successes": args.successes,
              "selected": [], "attempts": [], "shortfall": args.successes}

    def publish():
        temporary = args.output / ".selection.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "selection.json")

    publish()
    try:
        for index in range(args.attempts):
            episode = f"episode_{index:06d}"
            trajectory = args.output / "trajectories" / episode
            seed = args.seed + index
            command = [sys.executable, str(scripts / "rollout.py"), "--model", str(args.model),
                       "--assets", str(args.assets), "--output", str(trajectory),
                       "--steps", str(protocol.horizon_steps), "--seed", str(seed),
                       "--initial-index", str(index % pool.state_count),
                       "--initial-states", str(initial_states),
                       "--mode", args.mode, "--device", args.device]
            if args.full_state:
                command.append("--full-state")
            log_path = args.output / f"{episode}.log"
            with log_path.open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise RuntimeError(
                    f"{episode} exited with status {result.returncode}; see {log_path}"
                )
            scorer = "score_flatten.py" if task == "shirt_flatten" else "score_fold_trajectory.py"
            command = [sys.executable, str(scripts / scorer), str(trajectory),
                       "--template", str(args.template), "--protocol", str(protocol_path)]
            if task == "shirt_fold":
                command += ["--assets", str(args.assets)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                score_log = args.output / f"{episode}.score.log"
                score_log.write_text(result.stdout + result.stderr)
                raise RuntimeError(
                    f"{episode} scoring exited with status {result.returncode}; "
                    f"see {score_log}"
                )
            score = json.loads(result.stdout)
            (trajectory / "score.json").write_text(json.dumps(score, indent=2) + "\n")
            report["attempts"].append({"episode": episode, "seed": seed,
                                       "initial_index": index % pool.state_count,
                                       "success": score["success"]})
            if score["success"] is True:
                report["selected"].append(f"trajectories/{episode}")
            report["shortfall"] = max(0, args.successes - len(report["selected"]))
            publish()
            if report["shortfall"] == 0:
                break
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        publish()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
