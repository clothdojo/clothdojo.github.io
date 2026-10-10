"""Score a saved cloth state on CPU; no simulator or checkpoint is required."""

import argparse
import json
from pathlib import Path

import numpy as np

from clothdojo.evaluation.fold_quality import (
    FoldQualityMetric,
    prepare_fold_quality_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "state", nargs="?", type=Path,
        default=Path(__file__).parent / "data/fold_quality/expert_terminal.npz",
        help="NPZ with flat (N,3), triangles (M,3), and q (N,3) arrays",
    )
    args = parser.parse_args()
    with np.load(args.state, allow_pickle=False) as data:
        reference = prepare_fold_quality_reference(data["flat"], data["triangles"])
        metric = FoldQualityMetric.load()
        score = float(metric.score(data["q"], reference))
    print(json.dumps({
        "score": score,
        "threshold": metric.success_threshold,
        "frame_passes_threshold": bool(metric.passes_threshold(score)),
    }, indent=2))


if __name__ == "__main__":
    main()
