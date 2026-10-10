"""Compute Pi0.5 normalization with the reference training transforms."""

import argparse
import json
from pathlib import Path

from clothdojo.training.openpi import compute_norm_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-chunk-boundary", choices=("full_only", "repeat_last"),
                        default="full_only")
    args = parser.parse_args()
    result = compute_norm_stats(args.dataset, args.output,
                                action_chunk_boundary=args.action_chunk_boundary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
