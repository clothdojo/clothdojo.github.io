"""Download the three material sets used to re-render one trajectory and seed."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, nargs="?",
                        help="trajectory directory; omit with --selection")
    parser.add_argument("--selection", type=Path,
                        help="use the first successful trajectory in selection.json")
    parser.add_argument("--output", type=Path, default=Path("assets/visual"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.trajectory is None) == (args.selection is None):
        parser.error("provide a trajectory or --selection")
    if args.selection:
        from clothdojo.data.selection import first_selected_trajectory

        try:
            args.trajectory = first_selected_trajectory(args.selection)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    from clothdojo.assets.materials import load_material_catalog, prepare_polyhaven_assets
    from clothdojo.data.isaac_render import sample_visual_layout

    identity = hashlib.sha256((args.trajectory / "manifest.json").read_bytes()).hexdigest()
    layout = sample_visual_layout(args.seed, 0, source_episode_id=identity,
                                  randomize=("cloth", "table", "ground", "lighting"))
    catalog = load_material_catalog()
    selected = {layout.cloth.material, layout.table.material, layout.ground.material}
    subset = replace(catalog, materials=tuple(item for item in catalog.materials if item.id in selected),
                     source_files_hashes=tuple(item for item in catalog.source_files_hashes if item[0] in selected))
    result = prepare_polyhaven_assets(args.output / "polyhaven", subset)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
