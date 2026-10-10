"""Create an independent original-plus-mirror Piper trajectory collection."""

import argparse
import json
from pathlib import Path

from clothdojo.data.mirror import augment_mirror


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path,
                        help="directory containing trajectories/<episode>/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(augment_mirror(args.collection, args.output), indent=2))


if __name__ == "__main__":
    main()
