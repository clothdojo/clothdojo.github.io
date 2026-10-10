"""Score a saved flattening rollout against a canonical reference and hold rule."""

import argparse
import json
from pathlib import Path

from clothdojo.evaluation.scoring import score_trajectory
from clothdojo.trajectory import TrajectoryReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/evaluation/shirt_flatten.yaml"))
    args = parser.parse_args()
    if TrajectoryReader(args.trajectory).manifest.scene.task_name != "shirt_flatten":
        parser.error("expected a flattening trajectory and protocol")
    try:
        result = score_trajectory(args.trajectory, args.template, protocol_path=args.protocol)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
