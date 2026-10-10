"""Check the exact three-second folding witness in a recorded rollout."""

import argparse
import json
from pathlib import Path

from clothdojo.evaluation.scoring import score_trajectory
from clothdojo.trajectory import TrajectoryReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/evaluation/shirt_fold.yaml"))
    args = parser.parse_args()
    if TrajectoryReader(args.trajectory).manifest.scene.task_name != "shirt_fold":
        parser.error("expected a folding trajectory and protocol")
    try:
        result = score_trajectory(args.trajectory, args.template, assets=args.assets,
                                  protocol_path=args.protocol)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
