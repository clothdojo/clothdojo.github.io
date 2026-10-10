"""Export one recorded trajectory to a standard local LeRobot v2.1 dataset."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cloth-visual", choices=("nocs", "white"))
    args = parser.parse_args()
    from clothdojo.trajectory.lerobot import export_lerobot

    print(export_lerobot(args.trajectory, args.output, assets_root=args.assets,
                         device=args.device, cloth_visual=args.cloth_visual))


if __name__ == "__main__":
    main()
