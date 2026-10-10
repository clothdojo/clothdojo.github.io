"""Generate a settled canonical flattening target for one garment."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--garment", default="rand_ZZCWITH5HY")
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protocol", type=Path,
                        default=Path("configs/evaluation/shirt_flatten.yaml"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    from clothdojo.evaluation.templates import generate_asset_template

    result = generate_asset_template(args.garment, args.protocol, args.output,
                                     assets_root=args.assets, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
